// flashboot pipelined chain-broadcast engine (the ChainReceiver class), ported from
// the GB300-verified main-branch RingReceiver (src/fabric_broadcast.cu).
//
// Topology: seed(passive) -> clone0 -> clone1 -> ... -> cloneN-1. Every clone PULLS
// from its predecessor's fabric/ipc-mapped arena in fixed-size chunks, gated by a
// per-chunk u32 flag array the predecessor publishes as it lands each chunk:
//
//   per chunk j:  cuStreamWaitValue32(pred_flags[j] >= 1)   (skip when pred is the
//                                                            seed: its bytes are
//                                                            always valid)
//                 cudaMemcpyAsync(my_arena+off, pred_arena+off, n)   (copy engine)
//                 cuStreamWriteValue32(my_flags[j] = 1)      (publish to successor;
//                                                             DEFAULT inserts the
//                                                             barrier so the bytes
//                                                             are visible first)
//
// All hops therefore stream CONCURRENTLY and the chain completes in ~size/link_bw
// regardless of its length — no store-and-forward serialization, no doorbell
// messages on the control plane. Everything is hardware stream ops: no SMs beyond
// the tiny flag-reset kernel, the bulk bytes ride the copy engine.
//
// The successor imports the predecessor's ARENA and FLAGS with PeerArenaImporter
// (both are DeviceArena allocations, so both export ipc/fabric handles). The flags
// buffer MUST be zeroed before its handle is registered anywhere (fresh device
// memory is garbage, and garbage >= 1 releases a chunk early) — the Python side
// zeroes it via chain_zero_flags() before the rendezvous hello.
//
// Like the other engines this file is torch-free and NOT a standalone extension:
// register_chain_broadcast() is called from csrc/python_bindings.cc inside
// PYBIND11_MODULE(_C), shipping ChainReceiver + chain_zero_flags in flashboot._C.

#include <pybind11/pybind11.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

void runtime_check(cudaError_t error, const char* what) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("[fbchain] ") + what + " failed: " +
                             cudaGetErrorString(error));
}

void driver_check(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return;
  const char* error_string = nullptr;
  cuGetErrorString(result, &error_string);
  throw std::runtime_error(std::string("[fbchain] ") + what + " failed: " +
                           (error_string ? error_string : "unknown CUDA driver error"));
}

__global__ void fill_u32_kernel(uint32_t* ptr, uint32_t value, size_t count) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < count; i += stride) ptr[i] = value;
}

// One chain hop: a dedicated non-blocking stream that runs the wait/copy/publish
// loop. One ChainReceiver per clone rank, alive for the transfer only.
class ChainReceiver {
 public:
  explicit ChainReceiver(int device) : device_(device) {
    runtime_check(cudaSetDevice(device_), "cudaSetDevice");
    runtime_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                  "cudaStreamCreateWithFlags");
  }

  ~ChainReceiver() {
    if (stream_) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  ChainReceiver(const ChainReceiver&) = delete;
  ChainReceiver& operator=(const ChainReceiver&) = delete;

  // Pull [0, total_bytes) of the predecessor's arena into ours, chunk-pipelined.
  // pred_flags == 0 means the predecessor is the SEED (bytes always valid — no
  // waits). Blocking (synchronizes the stream); the GIL is released by the caller
  // binding. Chunk geometry MUST be uniform along the chain: my_flags has
  // ceil(total/chunk) entries and my successor indexes it with the same chunk size.
  void run(uintptr_t pred_arena, uintptr_t pred_flags, uintptr_t my_arena,
           uintptr_t my_flags, size_t total_bytes, size_t chunk_bytes) {
    if (!pred_arena || !my_arena || !my_flags)
      throw std::runtime_error("[fbchain] ChainReceiver.run: null buffer");
    if (chunk_bytes == 0)
      throw std::runtime_error("[fbchain] ChainReceiver.run: chunk_bytes == 0");

    runtime_check(cudaSetDevice(device_), "cudaSetDevice");
    const size_t nchunks = (total_bytes + chunk_bytes - 1) / chunk_bytes;

    // Re-zero my flags ON THE STREAM (ordered before every publish below). The
    // Python side already zeroed them before the rendezvous, so a successor that
    // races this reset only ever observes 0 -> blocks -> 1, never garbage.
    {
      const int threads = 256;
      const int blocks = (int)((nchunks + threads - 1) / threads);
      fill_u32_kernel<<<blocks > 0 ? blocks : 1, threads, 0, stream_>>>(
          reinterpret_cast<uint32_t*>(my_flags), 0u, nchunks);
    }

    const char* src = reinterpret_cast<const char*>(pred_arena);
    char* dst = reinterpret_cast<char*>(my_arena);
    for (size_t j = 0; j < nchunks; ++j) {
      const size_t off = j * chunk_bytes;
      const size_t n = (off + chunk_bytes <= total_bytes) ? chunk_bytes
                                                          : (total_bytes - off);
      if (pred_flags) {
        // Hardware stream wait — no kernel launch, the copy engine keeps streaming.
        driver_check(
            cuStreamWaitValue32(reinterpret_cast<CUstream>(stream_),
                                static_cast<CUdeviceptr>(pred_flags) + j * 4, 1u,
                                CU_STREAM_WAIT_VALUE_GEQ),
            "cuStreamWaitValue32(pred flag)");
      }
      runtime_check(cudaMemcpyAsync(dst + off, src + off, n,
                                    cudaMemcpyDeviceToDevice, stream_),
                    "cudaMemcpyAsync(chain chunk)");
      // DEFAULT inserts a memory barrier: the chunk's bytes are visible to the
      // successor before its flag flips.
      driver_check(
          cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_),
                               static_cast<CUdeviceptr>(my_flags) + j * 4, 1u,
                               CU_STREAM_WRITE_VALUE_DEFAULT),
          "cuStreamWriteValue32(my flag)");
    }
    runtime_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

 private:
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

// Synchronous device-memory u32 fill — zero a flags buffer BEFORE registering its
// handle anywhere (fresh allocations hold garbage, and garbage >= 1 would release
// chunks early on the successor).
void chain_zero_flags(uintptr_t flags, size_t count) {
  if (count == 0) return;
  const int threads = 256;
  const int blocks = (int)((count + threads - 1) / threads);
  fill_u32_kernel<<<blocks > 0 ? blocks : 1, threads>>>(
      reinterpret_cast<uint32_t*>(flags), 0u, count);
  runtime_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(zero flags)");
}

}  // namespace

// Registered INTO the main `_C` pybind module (see python_bindings.cc). Same GIL
// rule as the other engines: blocking native calls release the GIL via a LOCAL
// py::gil_scoped_release, never py::call_guard<>.
void register_chain_broadcast(py::module_& m) {
  py::class_<ChainReceiver>(m, "ChainReceiver")
      .def(py::init<int>(), py::arg("device"))
      .def(
          "run",
          [](ChainReceiver& receiver, uintptr_t pred_arena, uintptr_t pred_flags,
             uintptr_t my_arena, uintptr_t my_flags, size_t total_bytes,
             size_t chunk_bytes) {
            py::gil_scoped_release gil_off;  // multi-GB blocking pipeline
            receiver.run(pred_arena, pred_flags, my_arena, my_flags, total_bytes,
                         chunk_bytes);
          },
          py::arg("pred_arena"), py::arg("pred_flags"), py::arg("my_arena"),
          py::arg("my_flags"), py::arg("total_bytes"), py::arg("chunk_bytes"),
          "Chunk-pipelined pull of the predecessor arena (pred_flags=0 -> the "
          "predecessor is the seed, no waits). Blocking.");
  m.def("chain_zero_flags", &chain_zero_flags, py::arg("flags"), py::arg("count"),
        "Zero a per-chunk u32 flags buffer (call BEFORE registering its handle).");
}
