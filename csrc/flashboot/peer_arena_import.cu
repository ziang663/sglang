// flashboot handle-import clone engine (the PeerArenaImporter class).
//
// The clone IMPORTS the seed arena's 64-byte shareable handle (exported by DeviceArena::export_shareable_handle
// and carried over the TCP control plane), which maps the seed's arena into this
// process's address space. One plain cudaMemcpy device-to-device then pulls the whole
// arena over NVLink on the copy engine — no SMs, no bounce buffer, no QP.
//
// Two handle transports, identical flow (import -> copy -> release), differing only in
// the CUDA API family:
//   * "ipc"    — cudaIpcOpenMemHandle on a cudaIpcGetMemHandle blob. Same-node only
//     (the handle references a device allocation in another process on this machine).
//   * "fabric" — cuMemImportFromShareableHandle on a CUmemFabricHandle, then the VMM
//     reserve/map/set-access dance. Works across any GPUs in the same NVLink-fabric
//     IMEX domain (GB300 NVL72 racks), same node included.
//
// Like the other engines this file is torch-free and NOT a standalone extension:
// register_peer_import() is called from csrc/python_bindings.cc inside
// PYBIND11_MODULE(_C), so the class ships as `flashboot._C.PeerArenaImporter`.
//
// Build: part of the _C CUDAExtension — see setup.py SOURCES (links libcuda for cuMem*).

#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "flashboot/device_arena.h"  // fabric_granularity (same rounding as the exporter)

namespace py = pybind11;

namespace {

// Consume the failure we are about to report. A CUDA runtime call that fails LATCHES
// its error in the context, and the next thing to check cudaGetLastError() — inside
// torch, pages later, on an unrelated call — raises it as its own. That is not a
// cosmetic problem: an import that fails on purpose (an ipc handle from another node,
// which is how a clone discovers it must pull over the network instead) would poison
// every CUDA call after it, so the recovery path dies of an error that already had a
// handler. Clearing here keeps the throw the ONLY way this failure travels.
void clear_pending_cuda_error() { (void)cudaGetLastError(); }

void runtime_check(cudaError_t error, const char* what) {
  if (error == cudaSuccess) return;
  clear_pending_cuda_error();
  throw std::runtime_error(std::string("[fbpeer] ") + what + " failed: " +
                           cudaGetErrorString(error));
}

void driver_check(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return;
  clear_pending_cuda_error();   // same reason as runtime_check
  const char* error_string = nullptr;
  cuGetErrorString(result, &error_string);
  throw std::runtime_error(std::string("[fbpeer] ") + what + " failed: " +
                           (error_string ? error_string : "unknown CUDA driver error"));
}

size_t round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// One imported seed arena: the 64-byte handle mapped into this process, readable at
// peer_arena_pointer(). One PeerArenaImporter per clone rank; release() (or the
// destructor) drops the local mapping — the seed keeps owning the memory.
class PeerArenaImporter {
 public:
  PeerArenaImporter() = default;
  ~PeerArenaImporter() { release_mapping(); }

  PeerArenaImporter(const PeerArenaImporter&) = delete;
  PeerArenaImporter& operator=(const PeerArenaImporter&) = delete;

  // Map the seed arena behind `handle_bytes` (the raw 64-byte credential) into this
  // process, readable by `device`. `arena_bytes` must be the exporter's logical arena
  // size — the fabric path rounds it up with the SAME granularity the exporter used
  // (identical GPUs on both ends, which a byte-correct clone requires anyway).
  void import_handle(const std::string& transport, const std::string& handle_bytes,
                     size_t arena_bytes, int device) {
    py::gil_scoped_release gil_off;  // release the Python GIL during the blocking
    // native call. NOTE: the ipc open
    // can be slow when live serving processes occupy the GPUs involved, and it
    // cannot be overlapped in-process (the driver serializes this process's other
    // CUDA calls behind it). ipc-path-specific; the fabric transport imports via
    // cuMemImport/cuMemMap instead.
    if (imported_)
      throw std::runtime_error("[fbpeer] import_handle: already imported");
    runtime_check(cudaSetDevice(device), "cudaSetDevice");
    runtime_check(cudaFree(nullptr), "cudaFree(0)");  // ensure a primary context

    if (transport == "ipc") {
      import_ipc_handle(handle_bytes);
    } else if (transport == "fabric") {
      import_fabric_handle(handle_bytes, arena_bytes, device);
    } else {
      throw std::runtime_error("[fbpeer] unknown transport '" + transport +
                               "' (expected 'ipc' or 'fabric')");
    }
    transport_ = transport;
    arena_bytes_ = arena_bytes;
    device_ = device;
    imported_ = true;
  }

  // Local device pointer that maps the seed's arena (base + offset).
  uintptr_t peer_arena_pointer(size_t offset) const {
    if (!imported_)
      throw std::runtime_error("[fbpeer] peer_arena_pointer before import_handle");
    return reinterpret_cast<uintptr_t>(peer_base_) + offset;
  }

  // Pull [peer_offset, peer_offset+nbytes) of the seed arena into local GPU memory at
  // `local_destination`. Plain cudaMemcpy device-to-device: the driver routes it over
  // NVLink on the copy engine. D2D memcpy is async with respect to the host, so a
  // device synchronize makes this blocking like Endpoint.read.
  void copy_from_peer(uintptr_t local_destination, size_t peer_offset,
                      size_t nbytes) const {
    py::gil_scoped_release gil_off;  // multi-GB blocking copy
    if (!imported_)
      throw std::runtime_error("[fbpeer] copy_from_peer before import_handle");
    if (peer_offset + nbytes > arena_bytes_)
      throw std::runtime_error("[fbpeer] copy_from_peer: read past the seed arena end");
    runtime_check(cudaMemcpy(reinterpret_cast<void*>(local_destination),
                             reinterpret_cast<const char*>(peer_base_) + peer_offset,
                             nbytes, cudaMemcpyDeviceToDevice),
                  "cudaMemcpy(peer arena D2D)");
    runtime_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }

  // Drop the local mapping of the seed arena (idempotent; the destructor calls it
  // too). Purely local: the seed's allocation is untouched. The clone calls this right
  // after the pull so the mapped address range isn't held for the server's lifetime.
  void release() {
    py::gil_scoped_release gil_off;
    release_mapping();
  }

 private:
  void import_ipc_handle(const std::string& handle_bytes) {
    cudaIpcMemHandle_t ipc_handle{};
    if (handle_bytes.size() != sizeof(ipc_handle))
      throw std::runtime_error("[fbpeer] ipc handle must be " +
                               std::to_string(sizeof(ipc_handle)) + " bytes, got " +
                               std::to_string(handle_bytes.size()));
    std::memcpy(&ipc_handle, handle_bytes.data(), sizeof(ipc_handle));
    // Maps the seed's WHOLE allocation; peer access from the current device to the
    // seed's GPU is enabled lazily. Fails when seed and clone are not on the same
    // node — CUDA IPC is intra-node only (use fabric across nodes).
    cudaError_t error = cudaIpcOpenMemHandle(&peer_base_, ipc_handle,
                                             cudaIpcMemLazyEnablePeerAccess);
    if (error != cudaSuccess) {
      // The clone recovers from THIS one: a handle from another node (or a container
      // that shares neither the IPC namespace nor the GPUs) cannot be opened, and the
      // caller then pulls over the wire instead. So the context must be left clean.
      clear_pending_cuda_error();
      throw std::runtime_error(
          std::string("[fbpeer] cudaIpcOpenMemHandle failed: ") +
          cudaGetErrorString(error) +
          "\n  -> the ipc transport is same-node only: seed and clone must run on "
          "the same machine with peer-capable GPUs. Across nodes use the fabric "
          "transport (IMEX domain), or let the seed publish an rdma offer.");
    }
  }

  void import_fabric_handle(const std::string& handle_bytes, size_t arena_bytes,
                            int device) {
    CUmemFabricHandle fabric_handle{};
    if (handle_bytes.size() != sizeof(fabric_handle))
      throw std::runtime_error("[fbpeer] fabric handle must be " +
                               std::to_string(sizeof(fabric_handle)) + " bytes, got " +
                               std::to_string(handle_bytes.size()));
    std::memcpy(&fabric_handle, handle_bytes.data(), sizeof(fabric_handle));
    driver_check(cuMemImportFromShareableHandle(&fabric_allocation_handle_,
                                                &fabric_handle,
                                                CU_MEM_HANDLE_TYPE_FABRIC),
                 "cuMemImportFromShareableHandle");
    // Mirror the exporter's mapping: reserve a virtual range of the granularity-padded
    // size, map the imported allocation, grant this device access. READWRITE because
    // read-only access is not supported on all platforms (the clone never writes).
    mapped_bytes_ = round_up(arena_bytes, flashboot::fabric_granularity(device));
    try {
      CUdeviceptr virtual_base = 0;
      driver_check(cuMemAddressReserve(&virtual_base, mapped_bytes_, 0, 0, 0),
                   "cuMemAddressReserve(peer arena)");
      peer_base_ = reinterpret_cast<void*>(virtual_base);
      driver_check(cuMemMap(virtual_base, mapped_bytes_, 0,
                            fabric_allocation_handle_, 0),
                   "cuMemMap(peer arena)");
      CUmemAccessDesc access = {};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      driver_check(cuMemSetAccess(virtual_base, mapped_bytes_, &access, 1),
                   "cuMemSetAccess(peer arena)");
    } catch (...) {
      release_mapping();  // unwinds whatever partially succeeded (transport_ is not
      throw;              // "ipc" yet, so the fabric branch cleans up)
    }
  }

  void release_mapping() {
    if (!imported_ && !peer_base_ && !fabric_allocation_handle_) return;
    if (transport_ == "ipc") {
      if (peer_base_) cudaIpcCloseMemHandle(peer_base_);
    } else {
      if (peer_base_) {
        cuMemUnmap(reinterpret_cast<CUdeviceptr>(peer_base_), mapped_bytes_);
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(peer_base_), mapped_bytes_);
      }
      if (fabric_allocation_handle_) cuMemRelease(fabric_allocation_handle_);
    }
    peer_base_ = nullptr;
    fabric_allocation_handle_ = 0;
    mapped_bytes_ = 0;
    arena_bytes_ = 0;
    imported_ = false;
  }

  std::string transport_;                              // "ipc" or "fabric"
  void* peer_base_ = nullptr;                          // local mapping of the seed arena
  size_t arena_bytes_ = 0;                             // the seed's logical arena size
  size_t mapped_bytes_ = 0;                            // granularity-padded (fabric only)
  CUmemGenericAllocationHandle fabric_allocation_handle_ = 0;
  int device_ = -1;
  bool imported_ = false;
};

}  // namespace

// Registered INTO the main `_C` pybind module (see python_bindings.cc) — the class
// becomes `flashboot._C.PeerArenaImporter`.
void register_peer_import(py::module_& m) {
  // GIL rule: long/blocking native calls release the GIL via a LOCAL
  // py::gil_scoped_release inside the method body, never via py::call_guard<>
  // (call_guard broke GIL handling when _C coexists with torch's pybind11).
  py::class_<PeerArenaImporter>(m, "PeerArenaImporter")
      .def(py::init<>())
      .def("import_handle",
           [](PeerArenaImporter& importer, const std::string& transport,
              const py::bytes& handle_bytes, size_t arena_bytes, int device) {
             importer.import_handle(transport, std::string(handle_bytes), arena_bytes,
                                    device);
           },
           py::arg("transport"), py::arg("handle_bytes"), py::arg("arena_bytes"),
           py::arg("device"))
      .def("peer_arena_pointer", &PeerArenaImporter::peer_arena_pointer,
           py::arg("offset") = 0)
      .def("copy_from_peer", &PeerArenaImporter::copy_from_peer,
           py::arg("local_destination"), py::arg("peer_offset"), py::arg("nbytes"))
      .def("release", &PeerArenaImporter::release);
}
