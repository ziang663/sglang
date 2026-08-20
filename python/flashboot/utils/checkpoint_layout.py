"""ShardedStateLoader checkpoint -> aligned arena placement math (pure CPU).

A ShardedStateLoader checkpoint stores each rank's weights as
``model-rank-{rank}-part-{part}.safetensors`` files, already in their FINAL
post-``process_weights_after_loading`` layout. The flashboot arena lays those tensors
out in file order, bump-allocating each one at a 256-byte boundary:

  * WHY not the raw file layout: safetensors packs tensors back-to-back, so any
    odd-sized tensor (DeepSeek-V4's fp32 scalars like ``k_scale`` / ``hc_attn_scale``)
    knocks every later tensor down to 4-byte alignment. GPU kernels reject that —
    flashinfer's cute rmsnorm requires 16B-aligned tensor arguments, and DeepGEMM's TMA
    descriptors need 16-128B depending on swizzle. 256B (the CUDA-allocator grade)
    satisfies all of them.
  * WHY this stays fast: tensors are almost always 256B-multiples in size, so padding
    is rare (V4-Flash tp4: 260 pads, 56KB total over 47GB) and the arena remains long
    contiguous stretches of file bytes. Each such stretch is a :class:`CheckpointCopyRun`,
    so the disk->GPU move is still a few hundred big block copies, not one per tensor.

Seed and clone compute this layout independently from their own (identical) checkpoint
headers, so both sides agree on every offset and on the total arena size (the clone's
drift guard enforces the latter).

Everything here is deterministic metadata math over the safetensors headers — no torch
tensors, no GPU — so it is unit-testable offline. The GPU side consumes it via
:mod:`flashboot.arena` (rebind) and :mod:`flashboot.arena_fill` (fill).
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, NamedTuple, Tuple

from flashboot.utils.safetensors_metadata import (
    read_safetensors_header,
    safetensors_dtype_to_torch,
)

# Arena placement alignment for every tensor. 256B = what torch's caching allocator
# grants any fresh tensor, so no kernel can be pickier than this.
TENSOR_PLACEMENT_ALIGNMENT = 256

# A checkpoint exported in PADDED mode (sglang's ShardedStateLoader.save_model with
# pad_alignment) carries this alignment in the file already: the gaps are spelled out as
# real tensors, because safetensors requires the data region to be covered exactly and
# rejects holes. Such a tensor is not a model tensor — it occupies arena bytes so the
# NEXT one lands aligned, and nothing rebinds onto it.
PAD_KEY_SUFFIX = "!__pad__"


def is_padding_tensor(name: str) -> bool:
    """True for the filler an aligned export writes between real tensors."""
    return name.endswith(PAD_KEY_SUFFIX)


class TensorPlacement(NamedTuple):
    """One saved tensor's placement in the rank's arena + the metadata to rebind a model
    param/buffer onto it. Self-contained on purpose: by rebind time the model param's own
    storage has been dropped, so name/dtype/shape must all come from the checkpoint header
    (the arena holds raw bytes; dtype is the bytes' true format)."""
    name: str                # model state_dict key (a param or buffer)
    dtype: object            # torch.dtype from the checkpoint header
    shape: List[int]
    arena_offset: int        # byte offset of this tensor within the rank's arena
    nbytes: int


class CheckpointCopyRun(NamedTuple):
    """One contiguous stretch of checkpoint file bytes, and where it lands in the arena.

    A MAP, not a unit of work. Consecutive tensors merge into one run wherever no
    alignment padding separates them, so what the runs describe is the SHAPE OF THE
    SOURCE — which file bytes back which arena bytes. How the work is divided is a
    separate decision, made by :func:`flashboot.arena_fill.plan_fill_chunks`, which cuts
    the arena into uniform chunks and looks up the runs behind each one.

    Sizing the work by these runs instead would inherit the checkpoint's shape: a DSV4
    shard hands out 174 sub-megabyte runs alongside multi-gigabyte ones, and each tiny
    one still costs a full ping-pong cycle — sync, read, H2D, sync — to move a few
    hundred bytes. A padded export collapses this to one run per part file."""
    path: str
    file_offset: int         # absolute byte offset in the file where the run starts
    length: int              # bytes to copy
    arena_offset: int        # the run's base offset within the arena


def rank_part_files(checkpoint_dir: str, shard_rank: int) -> List[str]:
    """This rank's ``model-rank-{shard_rank}-part-*.safetensors``, sorted by part index."""
    pattern = os.path.join(checkpoint_dir, f"model-rank-{shard_rank}-part-*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"[flashboot] no shard files for rank {shard_rank}: {pattern} — is this a "
            f"ShardedStateLoader checkpoint (save_sharded_model output)?"
        )

    def part_index(path: str) -> int:
        m = re.search(r"-part-(\d+)\.safetensors$", path)
        return int(m.group(1)) if m else 0

    return sorted(files, key=part_index)


def align_up(offset: int, alignment: int = TENSOR_PLACEMENT_ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def read_checkpoint_layout(
    files: List[str],
) -> Tuple[List[TensorPlacement], List[CheckpointCopyRun], int]:
    """Parse this rank's part headers into
    ``(tensor_placements, copy_runs, total_arena_bytes)``.

    Tensors are placed in file order (parts in part-index order, tensors by their file
    data offset), each bump-allocated at :data:`TENSOR_PLACEMENT_ALIGNMENT`. Stretches
    with no padding in between collapse into single copy runs — which are a map from
    arena bytes to file bytes, NOT the unit the fill works in.
    """
    tensor_placements: List[TensorPlacement] = []
    copy_runs: List[CheckpointCopyRun] = []
    arena_cursor = 0
    for path in files:
        header, data_region_offset = read_safetensors_header(path)
        entries = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
        previous_file_end = None
        for name, info in entries:
            data_start, data_end = (int(x) for x in info["data_offsets"])
            nbytes = data_end - data_start
            # Filler goes exactly where it lies. Aligning IT would push it off the tensor
            # it follows and open the very gap it exists to close.
            arena_offset = (arena_cursor if is_padding_tensor(name)
                            else align_up(arena_cursor))
            run_continues = (copy_runs
                             and copy_runs[-1].path == path
                             and arena_offset == arena_cursor
                             and data_start == previous_file_end)
            if run_continues:
                copy_runs[-1] = copy_runs[-1]._replace(
                    length=copy_runs[-1].length + nbytes)
            else:
                copy_runs.append(CheckpointCopyRun(
                    path=path,
                    file_offset=data_region_offset + data_start,
                    length=nbytes,
                    arena_offset=arena_offset))
            if is_padding_tensor(name):
                # Consumes arena bytes (that is what keeps the real tensors aligned) but
                # is not a model tensor, so no placement and nothing to rebind.
                arena_cursor = arena_offset + nbytes
                previous_file_end = data_end
                continue
            tensor_placements.append(TensorPlacement(
                name=name,
                dtype=safetensors_dtype_to_torch(info["dtype"]),
                shape=list(info["shape"]),
                arena_offset=arena_offset,
                nbytes=nbytes,
            ))
            arena_cursor = arena_offset + nbytes
            previous_file_end = data_end
    return tensor_placements, copy_runs, arena_cursor


def clip_copy_runs(copy_runs: List[CheckpointCopyRun], start: int,
                   end: int) -> List[CheckpointCopyRun]:
    """The sub-runs of ``copy_runs`` whose ARENA bytes fall inside ``[start, end)`` —
    the DP-cooperative loader's disk diet: each replica feeds the fill pipeline only
    the runs (clipped at the segment boundary, file offsets shifted alike) that hit
    its native segment, so it reads exactly its share of the checkpoint bytes.
    Runs are arena-ordered in and stay arena-ordered out; empty intersections drop."""
    clipped: List[CheckpointCopyRun] = []
    for run in copy_runs:
        clip_start = max(run.arena_offset, int(start))
        clip_end = min(run.arena_offset + run.length, int(end))
        if clip_start >= clip_end:
            continue
        shift = clip_start - run.arena_offset
        clipped.append(CheckpointCopyRun(
            path=run.path,
            file_offset=run.file_offset + shift,
            length=clip_end - clip_start,
            arena_offset=clip_start))
    return clipped


def row_major_stride(shape: List[int]) -> List[int]:
    """C-contiguous (row-major) stride for ``shape`` — safetensors stores row-major.
    ``[] -> []`` (0-d scalar)."""
    if not shape:
        return []
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return stride
