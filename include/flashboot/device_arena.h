// flashboot — DeviceArena: one contiguous device-memory region per rank.
//
// The arena that holds a rank's whole weight image in checkpoint file layout. It backs
// both clone transports, which follow the same shape — the seed EXPORTS a per-arena
// credential, the clone IMPORTS it and pulls the bytes:
//
//   * ipc    (same node)   : plain cudaMalloc memory; the credential is the
//     64-byte cudaIpcGetMemHandle blob, which another process ON THE SAME NODE opens
//     and then copies from over NVLink with the copy engine.
//   * fabric (NVLink fabric): the arena must be allocated through the CUDA VMM API with
//     CU_MEM_HANDLE_TYPE_FABRIC (create(..., fabric_exportable=true)); the credential is
//     the 64-byte CUmemFabricHandle, importable by any GPU in the same IMEX domain
//     (e.g. a GB300 NVL72 rack), same node included.
//
// Usage (both seed and clone allocate their arena the same way):
//   DeviceArena arena;
//   arena.create(total_bytes, /*device=*/device);
//   void* base = arena.ptr(0);       // fill target / MR registration address
//   std::string handle = arena.export_shareable_handle("ipc");   // seed only
//
// The arena must stay alive as long as any model view aliases it AND as long as any
// clone may still import/read it (the Python side stashes the owning object on the
// model, which covers both).
#pragma once

#include <cstddef>
#include <string>

namespace flashboot {

// Allocation granularity of fabric-exportable VMM memory on `device`. Fabric arena
// sizes are rounded up to this; the import side (PeerArenaImporter) uses the same
// rounding to size its mapping of the exporter's allocation.
size_t fabric_granularity(int device);

class DeviceArena {
 public:
  DeviceArena() = default;
  ~DeviceArena();

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  // Allocate `size` bytes on `device`. Default: cudaMalloc, capacity() == size.
  // With fabric_exportable: CUDA VMM allocation with CU_MEM_HANDLE_TYPE_FABRIC
  // (required by export_shareable_handle("fabric")); capacity() is `size` rounded up
  // to the fabric allocation granularity. Throws std::runtime_error on failure.
  void create(size_t size, int device, bool fabric_exportable = false);

  // The 64-byte credential a clone imports to read this arena (PeerArenaImporter):
  //   transport == "ipc"    -> cudaIpcGetMemHandle    (needs a cudaMalloc arena)
  //   transport == "fabric" -> cuMemExportToShareableHandle (needs fabric_exportable)
  // Returned as raw bytes (the Python side hex-encodes it for the control plane).
  std::string export_shareable_handle(const std::string& transport) const;

  // Device pointer at `offset` bytes from the base (nullptr before create()).
  void* ptr(size_t offset = 0) const;

  size_t capacity() const { return capacity_; }
  int device() const { return device_; }

 private:
  void destroy();

  void* base_ = nullptr;
  size_t capacity_ = 0;
  int device_ = -1;
  bool created_ = false;
  bool fabric_exportable_ = false;
  // CUmemGenericAllocationHandle of the VMM allocation (fabric arenas only). Declared
  // as the underlying integer type so this header stays free of <cuda.h>.
  unsigned long long fabric_allocation_handle_ = 0;
};

}  // namespace flashboot
