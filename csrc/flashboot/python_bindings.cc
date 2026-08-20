// flashboot — pybind11 bindings (the flashboot._C extension).
//
// The Python layer works in terms of raw device pointers (int) so it can plug straight
// into torch.Tensor.data_ptr(); torch ownership stays on the Python side. This module
// exposes:
//   * DeviceArena       — the per-rank contiguous cudaMalloc weight arena
//   * copy_d2d          — blocking device-to-device copy (clone bounce -> arena)
//   * arena_view        — wrap a raw arena pointer as an aliasing torch.Tensor
//   * the NUMA-pinned host staging pipeline (alloc + parallel read + multi-stream H2D)
//     used by the seed's arena fill and the weight prestager
//   * PeerArenaImporter — the CUDA-IPC / NVLink-fabric handle-import clone engine
//                         (registered by peer_arena_import.cu)
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>  // for arena_view -> torch::Tensor (from_blob)

#include "flashboot/device_arena.h"
#include "flashboot/pinned_stage.h"

namespace py = pybind11;
using flashboot::DeviceArena;

namespace {

// Plain device-to-device copy between two raw pointers. Blocking. Runs on the copy
// engine (cudaMemcpy D2D), zero SM cost.
void copy_d2d(uintptr_t dst, uintptr_t src, size_t nbytes) {
  cudaError_t e = cudaMemcpy(reinterpret_cast<void*>(dst),
                             reinterpret_cast<const void*>(src), nbytes,
                             cudaMemcpyDeviceToDevice);
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("cudaMemcpy D2D failed: ") +
                             cudaGetErrorString(e));
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("cudaDeviceSynchronize failed: ") +
                             cudaGetErrorString(e));
}

}  // namespace

void register_peer_import(py::module_& m);  // PeerArenaImporter (peer_arena_import.cu)
void register_imex_check(py::module_& m);   // check_imex fabric preflight (imex_check.cc)
void register_chain_broadcast(py::module_& m);  // ChainReceiver pipeline (chain_broadcast.cu)
#ifndef FB_NO_RDMA
void register_rdma(py::module_& m);         // raw-ibverbs Endpoint (rdma_read.cpp)
#endif

PYBIND11_MODULE(_C, m) {
  m.doc() = "flashboot: sharded-checkpoint arena loader "
            "(ipc / fabric clone transports)";

  // --- the per-rank weight arena ---
  // The seed's exportable side of both clone transports: a 64-byte shareable handle
  // (ipc/fabric — fabric arenas must be created with fabric_exportable=True).
  py::class_<DeviceArena>(m, "DeviceArena")
      .def(py::init<>())
      .def("create", &DeviceArena::create, py::arg("size"), py::arg("device"),
           py::arg("fabric_exportable") = false)
      .def("export_shareable_handle",
           [](const DeviceArena& arena, const std::string& transport) {
             return py::bytes(arena.export_shareable_handle(transport));
           },
           py::arg("transport"),
           "The 64-byte credential a PeerArenaImporter opens: 'ipc' "
           "(cudaIpcGetMemHandle) or 'fabric' (cuMemExportToShareableHandle).")
      .def("ptr",
           [](const DeviceArena& arena, size_t offset) {
             return reinterpret_cast<uintptr_t>(arena.ptr(offset));
           },
           py::arg("offset") = 0)
      .def_property_readonly("capacity", &DeviceArena::capacity)
      .def_property_readonly("device", &DeviceArena::device);

  m.def("copy_d2d", &copy_d2d, py::arg("dst"), py::arg("src"), py::arg("nbytes"),
        "Blocking device-to-device cudaMemcpy between raw pointers (copy engine).");

  // Wrap a raw arena device pointer as a torch.Tensor that ALIASES that memory (no
  // copy, no ownership — the arena must outlive the tensor). Used to rebind model
  // parameters onto the arena so the model serves directly from it. dtype/device come
  // from `like`.
  m.def(
      "arena_view",
      [](uintptr_t ptr, std::vector<int64_t> shape, std::vector<int64_t> stride,
         const at::Tensor& like) {
        auto options = like.options().device(like.device());
        return torch::from_blob(
            reinterpret_cast<void*>(ptr), shape, stride,
            /*deleter=*/[](void*) {}, options);
      },
      py::arg("ptr"), py::arg("shape"), py::arg("stride"), py::arg("like"),
      "torch.Tensor aliasing arena memory at ptr with like's dtype/device.");

  // ===================== NUMA-pinned host -> GPU staging ======================
  // The seed fills its arena by reading this rank's checkpoint regions into a
  // NUMA-local pinned host buffer and copying that buffer to GPU.
  m.def(
      "copy_pinned_to_gpu",
      [](uintptr_t dst_gpu, uintptr_t src_host, size_t nbytes, int device,
         size_t chunk_bytes, int streams, bool register_in_chunks) {
        flashboot::H2DConfig cfg;
        cfg.device = device;
        cfg.chunk_bytes = chunk_bytes;
        cfg.streams = streams;
        cfg.register_in_chunks = register_in_chunks;
        flashboot::H2DStats stats;
        {
          // Release the GIL for the (blocking, multi-second) multi-stream H2D so the
          // concurrent CPU reader thread (the fill pump's prefetch of the next
          // sub-block) can actually run — without this the main thread holds the GIL
          // for the whole copy and read/H2D serialize instead of overlapping.
          py::gil_scoped_release release;
          stats = flashboot::copy_pinned_to_gpu(
              reinterpret_cast<void*>(dst_gpu),
              reinterpret_cast<void*>(src_host), nbytes, cfg);
        }
        py::dict d;
        d["total_bytes"] = stats.total_bytes;
        d["wall_s"] = stats.wall_s;
        d["gbps"] = stats.gbps;
        d["streams"] = stats.streams;
        d["registered_in_chunks"] = stats.registered_in_chunks;
        return d;
      },
      py::arg("dst_gpu"), py::arg("src_host"), py::arg("nbytes"),
      py::arg("device") = 0, py::arg("chunk_bytes") = (512ull << 20),
      py::arg("streams") = 4, py::arg("register_in_chunks") = false,
      "Copy pinned host bytes -> GPU via multi-stream cudaMemcpyAsync; returns "
      "timing dict. src_host must be CUDA-pinned unless register_in_chunks.");

  m.def(
      "alloc_pinned_on_node",
      [](size_t nbytes, int node) -> uintptr_t {
        void* p;
        {
          py::gil_scoped_release release;  // mbind + first-touch + cudaHostRegister block
          p = flashboot::alloc_pinned_on_node(nbytes, node);
        }
        return reinterpret_cast<uintptr_t>(p);
      },
      py::arg("nbytes"), py::arg("node"),
      "mmap+NUMA-bind+cudaHostRegister `nbytes` on `node`; returns the pinned host ptr.");

  m.def(
      "free_pinned_on_node",
      [](uintptr_t ptr, size_t nbytes) {
        py::gil_scoped_release release;
        flashboot::free_pinned_on_node(reinterpret_cast<void*>(ptr), nbytes);
      },
      py::arg("ptr"), py::arg("nbytes"),
      "cudaHostUnregister + munmap a buffer from alloc_pinned_on_node.");

  m.def(
      "read_file_into",
      [](const std::string& path, size_t file_off, size_t len, uintptr_t dst,
         size_t dst_off, int threads, size_t chunk, int numa_node) -> double {
        double wall_s;
        {
          py::gil_scoped_release release;  // blocking multi-thread mmap-memcpy / pread
          wall_s = flashboot::read_file_into(path, file_off, len,
                                             reinterpret_cast<void*>(dst), dst_off,
                                             threads, chunk, numa_node);
        }
        return wall_s;
      },
      py::arg("path"), py::arg("file_off"), py::arg("len"), py::arg("dst"),
      py::arg("dst_off") = 0, py::arg("threads") = 16,
      py::arg("chunk") = (16ull << 20), py::arg("numa_node") = -1,
      "Read [file_off,file_off+len) of path into dst+dst_off, threads-way parallel "
      "(tmpfs->mmap+memcpy, else pread64); worker threads bind to numa_node CPUs. "
      "Returns wall seconds.");

  m.def(
      "stage_file_to_shm",
      [](const std::string& src, const std::string& dst, int threads, size_t chunk,
         int numa_node) -> double {
        py::gil_scoped_release release;  // blocking parallel GPFS pread -> tmpfs mmap
        return flashboot::stage_file_to_shm(src, dst, threads, chunk, numa_node);
      },
      py::arg("src"), py::arg("dst"), py::arg("threads") = 8,
      py::arg("chunk") = (64ull << 20), py::arg("numa_node") = -1,
      "Copy a whole file GPFS->tmpfs by pread'ing straight into the mmap'd dst (one "
      "copy, no Python bytes bounce); threads bind to numa_node CPUs. Returns wall "
      "seconds.");

  m.def("path_in_ram", &flashboot::path_in_ram, py::arg("path"),
        "True if path resides on a RAM-backed filesystem (tmpfs/ramfs, e.g. /dev/shm), "
        "via statfs f_type; reads then hit page cache at memcpy speed.");

  register_peer_import(m);  // flashboot._C.PeerArenaImporter (ipc/fabric handle import)
  register_imex_check(m);   // flashboot._C.check_imex (fabric/IMEX preflight)
  register_chain_broadcast(m);  // flashboot._C.ChainReceiver (pipelined chain broadcast)
#ifndef FB_NO_RDMA
  register_rdma(m);         // flashboot._C.Endpoint (raw-ibverbs rdma collectives)
#endif
}
