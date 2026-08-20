// flashboot — pinned_stage: NUMA-local host staging for the arena fill.
//
// The sharded arena loader fills its GPU arena by reading this rank's checkpoint
// regions into a NUMA-pinned host buffer and copying that buffer to the GPU. Two
// independent, composable pieces (pure bytes — no torch/tensor types):
//
//   alloc_pinned_on_node(nbytes, node)   // NUMA-bound, cudaHostRegister'd buffer
//   read_file_into(path, ..., dst)  // parallel disk/tmpfs read into the buffer
//   copy_pinned_to_gpu(gpu, host, total) // multi-stream H2D
//
// NUMA-local throughout: the read threads AND the pinned buffer sit on the GPU's
// CPU socket (gpu0-3->node0, gpu4-7->node1 on H100), so neither the src->pinned
// read nor the pinned->GPU H2D crosses the inter-socket link.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace flashboot {

// ===================== host(pinned) -> GPU ======================
struct H2DConfig {
  int device = 0;
  size_t chunk_bytes = 512ull << 20;
  int streams = 4;
  // If src_host is NOT pre-registered (e.g. a freshly-attached shm), set this:
  // the copy registers src_host in chunk_bytes pieces and overlaps the
  // cudaHostRegister of chunk N+1 with the H2D of chunk N, hiding registration
  // latency behind the transfer. If false, src_host MUST already be pinned.
  bool register_in_chunks = false;
};
struct H2DStats {
  size_t total_bytes = 0;
  double wall_s = 0, gbps = 0;
  int streams = 0;
  bool registered_in_chunks = false;
};
// Copy [src_host, src_host+nbytes) -> dst_gpu via `streams` concurrent streams.
H2DStats copy_pinned_to_gpu(void* dst_gpu, void* src_host, size_t nbytes,
                            const H2DConfig& cfg);

// ===================== NUMA / in-RAM helpers ==================
// True if `path` lives on a tmpfs/ramfs (RAM-backed) FS — statfs(2) f_type magic
// One syscall, no /proc parsing.
bool path_in_ram(const std::string& path);

// mmap(MAP_PRIVATE|ANON) `nbytes`, NUMA-bind to `node` (mbind, when node>=0),
// first-touch memset (realize placement), then cudaHostRegister so it's a valid
// pinned H2D source. Returns the base pointer; throws std::runtime_error on failure.
// torch's pin_memory ignores set_mempolicy, so this is how we force the socket.
void* alloc_pinned_on_node(size_t nbytes, int node);

// cudaHostUnregister + munmap a buffer from alloc_pinned_on_node (best-effort).
void free_pinned_on_node(void* ptr, size_t nbytes);

// Read [file_off, file_off+len) of `path` into `dst`+`dst_off`, `threads`-way
// parallel in `chunk`-byte pieces. tmpfs/ramfs source -> mmap(MAP_PRIVATE, cached
// process-wide, NEVER munmap'd on the hot path) + memcpy (bypasses the per-read
// kernel copy_to_user that degrades under N-rank concurrency); disk/network source
// -> pread64. Worker threads bind to `numa_node`'s CPUs when node>=0 (NUMA-local
// memcpy). Returns wall seconds. Subsumes the old python mmap/preadv read branches.
double read_file_into(const std::string& path, size_t file_off, size_t len,
                           void* dst, size_t dst_off, int threads, size_t chunk,
                           int numa_node);

// Copy a whole file `src` (e.g. a GPFS shard) to `dst` on a tmpfs (e.g. /dev/shm), by
// mmap'ing dst and pread'ing `src` directly into the mapped tmpfs pages — one kernel copy,
// no userspace bounce buffer or per-chunk allocation (the Python os.pread/os.pwrite path
// costs two copies + a bytes object per chunk under the GIL, ~1/3 the throughput). `threads`
// pread in parallel, bound to `numa_node`'s CPUs when node>=0 so the tmpfs pages land local.
// The caller is responsible for atomic publish (write to a temp name, then rename). Returns
// wall seconds. Throws std::runtime_error on failure.
double stage_file_to_shm(const std::string& src, const std::string& dst, int threads,
                         size_t chunk, int numa_node);

}  // namespace flashboot
