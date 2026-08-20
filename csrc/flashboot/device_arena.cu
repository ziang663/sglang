#include "flashboot/device_arena.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace flashboot {
namespace {

void cuda_check(cudaError_t error, const char* what) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(what) + " failed: " +
                             cudaGetErrorString(error));
}

// Driver-API (cuMem*) variant of cuda_check, used by the fabric VMM path. The
// NOT_PERMITTED case gets an IMEX hint: fabric handles only work when exporter and
// importer share an IMEX channel (nvidia-caps-imex-channels), which plain H100 nodes
// don't have — use the ipc transport for same-node clones there.
void driver_check(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return;
  const char* error_string = nullptr;
  cuGetErrorString(result, &error_string);
  std::string message = std::string(what) + " failed: " +
                        (error_string ? error_string : "unknown CUDA driver error");
  if (result == CUDA_ERROR_NOT_PERMITTED || result == CUDA_ERROR_NOT_SUPPORTED) {
    message += "\n  -> fabric memory needs an NVLink-fabric / IMEX-configured system "
               "(see /dev/nvidia-caps-imex-channels/). For a same-node clone on plain "
               "H100 use FLASHBOOT_TRANSPORT=ipc instead.";
  }
  throw std::runtime_error(message);
}

// Ensure a primary context exists on `device` (cudaMalloc / cudaIpc* / cuMem* need one).
void bind_device(int device) {
  cuda_check(cudaSetDevice(device), "cudaSetDevice");
  cuda_check(cudaFree(nullptr), "cudaFree(0)");
}

// Allocation properties of a fabric-exportable VMM allocation on `device`. Shared by
// the arena (export side) and PeerArenaImporter (import side, for the granularity).
CUmemAllocationProp fabric_allocation_properties(int device) {
  CUmemAllocationProp properties = {};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  return properties;
}

size_t round_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

size_t fabric_granularity(int device) {
  CUmemAllocationProp properties = fabric_allocation_properties(device);
  size_t granularity = 0;
  driver_check(cuMemGetAllocationGranularity(&granularity, &properties,
                                             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
               "cuMemGetAllocationGranularity(fabric)");
  return granularity;
}

void DeviceArena::create(size_t size, int device, bool fabric_exportable) {
  if (created_) throw std::runtime_error("DeviceArena already created");
  if (size == 0) throw std::runtime_error("DeviceArena.create: size must be > 0");

  bind_device(device);
  device_ = device;
  fabric_exportable_ = fabric_exportable;

  if (!fabric_exportable) {
    cuda_check(cudaMalloc(&base_, size), "cudaMalloc(arena)");
    capacity_ = size;
    created_ = true;
    return;
  }

  // Fabric path: physical VMM allocation (exportable as a CUmemFabricHandle) mapped
  // into a fresh virtual-address range. Sizes must be granularity-multiples.
  const size_t padded_size = round_up(size, fabric_granularity(device));
  created_ = true;  // from here on destroy() can unwind whatever partially succeeded
  try {
    CUmemAllocationProp properties = fabric_allocation_properties(device);
    driver_check(cuMemCreate(&fabric_allocation_handle_, padded_size, &properties, 0),
                 "cuMemCreate(fabric arena)");
    CUdeviceptr virtual_base = 0;
    driver_check(cuMemAddressReserve(&virtual_base, padded_size, 0, 0, 0),
                 "cuMemAddressReserve(fabric arena)");
    base_ = reinterpret_cast<void*>(virtual_base);
    capacity_ = padded_size;  // destroy() needs the reserved length from here on
    driver_check(cuMemMap(virtual_base, padded_size, 0, fabric_allocation_handle_, 0),
                 "cuMemMap(fabric arena)");
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    driver_check(cuMemSetAccess(virtual_base, padded_size, &access, 1),
                 "cuMemSetAccess(fabric arena)");
  } catch (...) {
    destroy();
    throw;
  }
}

std::string DeviceArena::export_shareable_handle(const std::string& transport) const {
  if (!created_)
    throw std::runtime_error("DeviceArena.export_shareable_handle: not created");

  if (transport == "ipc") {
    if (fabric_exportable_)
      throw std::runtime_error(
          "DeviceArena: ipc export needs a cudaMalloc arena, but this arena was "
          "created fabric_exportable (cudaIpcGetMemHandle rejects VMM memory)");
    cudaIpcMemHandle_t ipc_handle{};
    cuda_check(cudaIpcGetMemHandle(&ipc_handle, base_), "cudaIpcGetMemHandle(arena)");
    return std::string(reinterpret_cast<const char*>(&ipc_handle), sizeof(ipc_handle));
  }

  if (transport == "fabric") {
    if (!fabric_exportable_)
      throw std::runtime_error(
          "DeviceArena: fabric export needs create(..., fabric_exportable=true) — "
          "this arena is plain cudaMalloc memory");
    CUmemFabricHandle fabric_handle{};
    driver_check(cuMemExportToShareableHandle(&fabric_handle, fabric_allocation_handle_,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0),
                 "cuMemExportToShareableHandle(arena)");
    return std::string(reinterpret_cast<const char*>(&fabric_handle),
                       sizeof(fabric_handle));
  }

  throw std::runtime_error("DeviceArena.export_shareable_handle: unknown transport '" +
                           transport + "' (expected 'ipc' or 'fabric')");
}

void* DeviceArena::ptr(size_t offset) const {
  if (!created_) return nullptr;
  return reinterpret_cast<void*>(reinterpret_cast<char*>(base_) + offset);
}

void DeviceArena::destroy() {
  if (!created_) return;
  if (fabric_exportable_) {
    if (base_) {
      cuMemUnmap(reinterpret_cast<CUdeviceptr>(base_), capacity_);
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_), capacity_);
    }
    if (fabric_allocation_handle_) cuMemRelease(fabric_allocation_handle_);
  } else if (base_) {
    cudaFree(base_);
  }
  base_ = nullptr;
  capacity_ = 0;
  fabric_allocation_handle_ = 0;
  fabric_exportable_ = false;
  created_ = false;
}

DeviceArena::~DeviceArena() { destroy(); }

}  // namespace flashboot
