"""Disk/tmpfs -> pinned host -> GPU arena fill (the seed's weight move).

The fill streams the arena through a 2-slot NUMA-local pinned ping-pong buffer: while the
multi-stream device copy of chunk k drains, a background thread reads chunk k+1 into the
OTHER slot, so the read and the copy overlap. The OS-level pieces (parallel pread/mmap
read, NUMA-pinned allocation, thread CPU binding) are native in ``flashboot._C``.

**The unit of work is a uniform span of ARENA bytes, not a tensor and not a copy run.**
That distinction is the whole design of this module. The checkpoint's copy runs
(:class:`flashboot.utils.checkpoint_layout.CheckpointCopyRun`) say where bytes come FROM; they
are a map, and letting them size the work would inherit the checkpoint's shape. A DSV4
shard produces 174 sub-megabyte runs alongside multi-gigabyte ones, and every tiny one
still costs a full ping-pong cycle — sync, read, H2D, sync — to move a few hundred bytes.
Sizing the unit from the PARALLELISM instead (one knob, ``FB_FILL_CHUNK_MB``) makes every
step cost the same whatever the checkpoint looks like: DSV4 went from 348 units to 88
uniform ones, 2.35s to 2.20s, and its run-to-run spread from 7.15/2.57/2.49 to
2.31/2.40/2.32. Qwen, whose tensors are all 256-byte multiples, was 29 either way and
showed none of this.

A chunk resolves to one read, or two where it straddles a part-file boundary. A chunk with
no source at all is dropped; the alignment holes INSIDE a chunk are zeroed, because the
chunk is copied in one operation and a hole would otherwise carry whatever the slot held
two chunks ago — see :meth:`ArenaFillChunk.unbacked_spans`.

All tunables are read from the environment ONCE into :class:`ArenaFillConfig` so the hot
loop never touches ``os.getenv`` and every knob is discoverable in one place.
"""
from __future__ import annotations

import ctypes
import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, NamedTuple, Tuple

from flashboot import _C
from flashboot.utils.checkpoint_layout import CheckpointCopyRun

_MB = 1 << 20


class ChunkFileRead(NamedTuple):
    """One contiguous file->pinned-slot read inside a chunk."""
    path: str
    file_offset: int
    slot_offset: int         # where in the pinned slot this piece lands
    length: int


class ArenaFillChunk(NamedTuple):
    """One UNIFORM slice of the arena, and the file reads that fill it.

    ``arena_offset`` is an offset WITHIN the arena, not an absolute address, so a plan can
    be inspected without a GPU. ``length`` is the uniform chunk size except for the tail.
    """
    arena_offset: int
    length: int
    reads: List[ChunkFileRead]

    def unbacked_spans(self) -> List[Tuple[int, int]]:
        """``(slot_offset, length)`` of the bytes inside this chunk that NO read fills —
        the alignment padding an unpadded checkpoint needs between tensors.

        The chunk is copied to the device in one operation, so these bytes are copied
        too, whatever the slot happened to hold. Zeroing them first is what keeps the
        arena a deterministic function of the checkpoint: without it a hole picks up
        whatever the chunk that used this slot two steps ago left there, so two fills of
        the same checkpoint would not checksum alike. They are cheap to zero — a DSV4
        shard's holes come to 56KB over 47GB — and no tensor is ever bound onto them.
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for read in sorted(self.reads, key=lambda r: r.slot_offset):
            if read.slot_offset > cursor:
                spans.append((cursor, read.slot_offset - cursor))
            cursor = max(cursor, read.slot_offset + read.length)
        if cursor < self.length:
            spans.append((cursor, self.length - cursor))
        return spans


@dataclasses.dataclass
class ArenaFillConfig:
    """Fill tunables (one env read at loader construction).

    NUMA: read threads + the pinned buffer bind to the GPU's CPU socket (gpu0-3 ->
    node0, gpu4-7 -> node1 on H100) so neither the source->pinned read nor the pinned->GPU
    copy crosses the inter-socket link. The device chunk/stream defaults are the swept
    optimum (64MB chunks, 8 streams — see experiment_logs/glm51-h2d-pin-sweep).
    """
    numa_bind: bool
    gpus_per_numa_node: int
    pinned_slot_capacity: int   # bytes per ping-pong slot (2 slots allocated)
    fill_chunk_bytes: int       # the work unit; 0 = derive it from the pinned slot
    h2d_chunk_bytes: int
    h2d_streams: int
    read_workers: int
    read_chunk_bytes: int

    @classmethod
    def from_environment(cls) -> "ArenaFillConfig":
        return cls(
            numa_bind=os.getenv("FB_FLASHSHARDED_NUMA_BIND", "1") != "0",
            gpus_per_numa_node=int(os.getenv("FB_FLASHSHARDED_GPUS_PER_NUMA", "4")),
            pinned_slot_capacity=int(os.getenv("FB_FLASHSHARDED_PIN_MB", "512")) * _MB,
            fill_chunk_bytes=int(os.getenv("FB_FILL_CHUNK_MB", "0")) * _MB,
            h2d_chunk_bytes=int(os.getenv("SGLANG_FABRIC_FLASH_H2D_CHUNK_MB", "64")) * _MB,
            h2d_streams=int(os.getenv("FB_FLASHSHARDED_H2D_STREAMS", "8")),
            read_workers=int(os.getenv("FB_FLASHSHARDED_READ_WORKERS", "16")),
            read_chunk_bytes=int(os.getenv("FB_FLASHSHARDED_READ_CHUNK_MB", "16")) * _MB,
        )

    def numa_node_for_gpu(self, gpu: int) -> int:
        """NUMA node (CPU socket) for this GPU's read threads + pinned buffer, or -1 to
        disable binding. Do NOT use the GPU's PCI numa_node (a per-rank process can
        report the GPU-side node 2/3, not the CPU socket) — use the verified
        4-GPUs-per-socket layout."""
        if not self.numa_bind:
            return -1
        return (gpu % 8) // max(self.gpus_per_numa_node, 1)

    def chunk_bytes_for(self, span_bytes: int) -> int:
        """The uniform work-unit size for a span of ``span_bytes``.

        ``FB_FILL_CHUNK_MB`` decides it; unset, the pinned slot does. Never below one
        device-copy chunk (there would be nothing to spread across the copy streams) and
        never above the span itself (a single short chunk beats one padded slot)."""
        chunk_bytes = self.fill_chunk_bytes
        if chunk_bytes <= 0:
            chunk_bytes = max(self.pinned_slot_capacity, self.h2d_chunk_bytes)
        return min(max(chunk_bytes, self.h2d_chunk_bytes), max(span_bytes, 1))


def plan_fill_chunks(copy_runs: List[CheckpointCopyRun],
                     config: ArenaFillConfig) -> Tuple[List[ArenaFillChunk], int]:
    """Cut the arena into UNIFORM chunks and resolve which file bytes back each one.

    Deliberately blind to tensors, and deliberately blind to the runs' own boundaries: the
    runs are only consulted to answer "what backs these arena bytes".

    The span covered is derived from the runs — from the first byte any run places to the
    last — so a caller that hands over a SUBSET (a data-parallel replica passing only the
    runs clipped to its own segment) automatically chunks over exactly that segment and
    nothing else. Chunks with no file bytes behind them are dropped rather than emitted,
    so the pump can never copy a slot it did not fill: arena bytes with no source (the
    alignment padding an unpadded checkpoint needs, or another replica's segment) are
    simply left alone.

    Returns ``(chunks, chunk_bytes)``.
    """
    if not copy_runs:
        return [], 0
    ordered = sorted(copy_runs, key=lambda run: run.arena_offset)
    span_start = ordered[0].arena_offset
    span_end = max(run.arena_offset + run.length for run in ordered)
    chunk_bytes = config.chunk_bytes_for(span_end - span_start)

    chunks: List[ArenaFillChunk] = []
    cursor = span_start
    first_live_run = 0
    while cursor < span_end:
        length = min(chunk_bytes, span_end - cursor)
        chunk_end = cursor + length
        # Runs are arena-ordered, so anything ending at or before this chunk can never
        # back a later one either — advance the window instead of rescanning.
        while (first_live_run < len(ordered)
               and ordered[first_live_run].arena_offset
               + ordered[first_live_run].length <= cursor):
            first_live_run += 1
        reads: List[ChunkFileRead] = []
        probe = first_live_run
        while probe < len(ordered) and ordered[probe].arena_offset < chunk_end:
            run = ordered[probe]
            overlap_start = max(cursor, run.arena_offset)
            overlap_end = min(chunk_end, run.arena_offset + run.length)
            if overlap_end > overlap_start:
                reads.append(ChunkFileRead(
                    path=run.path,
                    file_offset=run.file_offset + (overlap_start - run.arena_offset),
                    slot_offset=overlap_start - cursor,
                    length=overlap_end - overlap_start))
            probe += 1
        if reads:
            chunks.append(ArenaFillChunk(arena_offset=cursor, length=length,
                                         reads=reads))
        cursor = chunk_end
    return chunks, chunk_bytes


def _fill_double_buffered(chunks: List[ArenaFillChunk], chunk_bytes: int,
                          pinned_ptr: int, arena_base: int, gpu: int, have_cuda: bool,
                          numa_node: int, config: ArenaFillConfig) -> None:
    """Read/copy each chunk with a 2-slot ping-pong: while the device copy of chunk k
    drains on the main thread (``copy_pinned_to_gpu`` releases the GIL), one background
    thread reads chunk k+1 into the OTHER slot (``read_file_into`` also releases the
    GIL)."""
    import torch  # deferred: this module is imported by CPU-only unit tests

    def read_chunk_into_slot(chunk: ArenaFillChunk, slot_index: int) -> float:
        # tmpfs -> mmap(cached)+memcpy, disk -> pread; worker threads bind to numa_node.
        # A chunk has one read, or two where it straddles a part-file boundary.
        slot_base = slot_index * chunk_bytes
        # Alignment holes first: the whole chunk is copied in one go, so a hole would
        # otherwise carry whatever this slot held two chunks ago (see unbacked_spans).
        for hole_offset, hole_length in chunk.unbacked_spans():
            ctypes.memset(pinned_ptr + slot_base + hole_offset, 0, hole_length)
        return sum(_C.read_file_into(read.path, read.file_offset, read.length,
                                     pinned_ptr, slot_base + read.slot_offset,
                                     config.read_workers, config.read_chunk_bytes,
                                     numa_node)
                   for read in chunk.reads)

    # Profiling accumulators: io_wait = main-thread time blocked on the next chunk's read
    # future (>0 means the read cannot keep up with the copy -> read/storage-bound);
    # h2d_time = time in copy_pinned_to_gpu; read_wall_sum overlaps io_wait.
    io_wait = 0.0
    h2d_time = 0.0
    read_wall_sum = 0.0
    bytes_filled = 0
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as reader:
        # prefetch chunk 0 into slot 0; the loop copies chunk k while reading chunk k+1.
        read_future = reader.submit(read_chunk_into_slot, chunks[0], 0)
        for chunk_index, chunk in enumerate(chunks):
            current_slot = chunk_index % 2
            if have_cuda:
                torch.cuda.synchronize(gpu)     # drain the prev copy that used this slot
            waited_from = time.perf_counter()
            read_wall_sum += read_future.result()   # wait this chunk's read (returns wall)
            io_wait += time.perf_counter() - waited_from
            if chunk_index + 1 < len(chunks):       # prefetch the next chunk, OTHER slot
                read_future = reader.submit(read_chunk_into_slot,
                                            chunks[chunk_index + 1], current_slot ^ 1)
            h2d_from = time.perf_counter()
            _C.copy_pinned_to_gpu(arena_base + chunk.arena_offset,
                                  pinned_ptr + current_slot * chunk_bytes,
                                  chunk.length, gpu, config.h2d_chunk_bytes,
                                  config.h2d_streams, False)
            if have_cuda:
                torch.cuda.synchronize(gpu)
            h2d_time += time.perf_counter() - h2d_from
            bytes_filled += chunk.length

    wall = time.perf_counter() - wall_start
    # RAM-backed source? tmpfs/ramfs (e.g. a /dev/shm prestage) reads at memcpy speed.
    first_path = next((read.path for chunk in chunks for read in chunk.reads), None)
    source_in_ram = _C.path_in_ram(str(first_path)) if first_path else False
    print(f"[flashboot][fill] gpu={gpu} chunks={len(chunks)}x{chunk_bytes >> 20}MB "
          f"GB={bytes_filled / 1e9:.1f} src_in_ram={source_in_ram} wall={wall:.2f}s "
          f"io_wait={io_wait:.2f}s h2d={h2d_time:.2f}s read_sum={read_wall_sum:.2f}s "
          f"read_bw={bytes_filled / 1e9 / max(read_wall_sum, 1e-6):.1f}GB/s "
          f"({'READ/GPFS-bound' if io_wait > h2d_time else 'H2D-bound'})", flush=True)


def fill_arena_from_checkpoint(copy_runs: List[CheckpointCopyRun], arena_base: int,
                               gpu: int, have_cuda: bool,
                               config: ArenaFillConfig) -> None:
    """Fill the arena bytes that ``copy_runs`` place, and only those.

    Cuts the span the runs cover into uniform chunks, allocates the 2-slot NUMA-pinned
    ping-pong buffer (``_C.alloc_pinned_on_node``: mmap + mbind + cudaHostRegister —
    torch's pin_memory ignores mempolicy), pumps, and frees.

    A data-parallel replica passes only the runs clipped to its own segment, so it reads
    and writes exactly its share; no separate bound is needed.
    """
    import torch

    numa_node = config.numa_node_for_gpu(gpu)
    chunks, chunk_bytes = plan_fill_chunks(copy_runs, config)
    if not chunks:
        return
    pinned_bytes = 2 * chunk_bytes
    pinned_ptr = _C.alloc_pinned_on_node(pinned_bytes, numa_node)
    try:
        _fill_double_buffered(chunks, chunk_bytes, pinned_ptr, arena_base, gpu,
                              have_cuda, numa_node, config)
        if have_cuda:
            torch.cuda.synchronize(gpu)
    finally:
        _C.free_pinned_on_node(pinned_ptr, pinned_bytes)
