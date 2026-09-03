#include "nn/cuda_util.h"

#include "util/exception.h"

#include <cuda_runtime_api.h>
#include <format>
#include <string>

namespace scribblez {
namespace nn {

namespace {

// Throw on a non-success status, naming the CUDA op. These are genuinely
// unexpected failures (a misconfigured GPU, OOM); util::Exception lets the
// top-level handler surface the message rather than exiting silently.
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw util::Exception("CUDA error in {}: {}", what, cudaGetErrorString(err));
  }
}

// The current device's properties (compute capability, name, limits).
cudaDeviceProp current_device_props() {
  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return prop;
}

}  // namespace

const char* sm_tag() {
  // Function-local static: C++ guarantees the initializer runs exactly once,
  // even under concurrent first calls.
  static const std::string tag = [] {
    const cudaDeviceProp prop = current_device_props();
    return std::format("{}.{}", prop.major, prop.minor);
  }();
  return tag.c_str();
}

int compute_capability_major() { return current_device_props().major; }

void set_device(int device_id) { check(cudaSetDevice(device_id), "cudaSetDevice"); }

stream_t create_stream() {
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  return stream;
}

void destroy_stream(stream_t stream) { check(cudaStreamDestroy(stream), "cudaStreamDestroy"); }

void synchronize_stream(stream_t stream) {
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

size_t device_memory_used() {
  size_t free_bytes = 0, total_bytes = 0;
  check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  return total_bytes - free_bytes;
}

void* device_malloc(size_t n_bytes) {
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, n_bytes), "cudaMalloc");
  return ptr;
}

void device_free(void* ptr) { check(cudaFree(ptr), "cudaFree"); }

void* host_malloc(size_t n_bytes) {
  void* ptr = nullptr;
  check(cudaMallocHost(&ptr, n_bytes), "cudaMallocHost");
  return ptr;
}

void host_free(void* ptr) { check(cudaFreeHost(ptr), "cudaFreeHost"); }

void host_to_device_async(stream_t stream, void* dst, const void* src, size_t n_bytes) {
  check(cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyHostToDevice, stream),
        "cudaMemcpyAsync(HostToDevice)");
}

void device_to_host_async(stream_t stream, void* dst, const void* src, size_t n_bytes) {
  check(cudaMemcpyAsync(dst, src, n_bytes, cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(DeviceToHost)");
}

}  // namespace nn
}  // namespace scribblez
