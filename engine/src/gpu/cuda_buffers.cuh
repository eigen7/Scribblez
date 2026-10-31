#pragma once

// Memory for the CUDA translation units: error checks, a table's one-time
// upload, and the per-call buffers, device and page-locked host, that grow to
// the largest call.

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace scribblez::gpu {

// Throws std::runtime_error on a CUDA failure.
inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
  }
}

// A copy of `v` in device memory, its size added to `bytes`.
template <typename T>
T* upload(const std::vector<T>& v, size_t* bytes) {
  T* d = nullptr;
  const size_t n = v.size() * sizeof(T);
  cuda_check(cudaMalloc(&d, n == 0 ? 1 : n), "malloc");
  if (n) cuda_check(cudaMemcpy(d, v.data(), n, cudaMemcpyHostToDevice), "memcpy");
  *bytes += n;
  return d;
}

// A device array that only grows.
template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t cap = 0;

  void reserve(size_t n);
  // Copies v to the device, growing as needed.
  void upload(const T* v, size_t n);
  ~DeviceBuffer() { cudaFree(data); }
};

// Page-locked host memory that only grows: what cudaMemcpyAsync needs to run
// asynchronously.
template <typename T>
struct PinnedBuffer {
  T* data = nullptr;
  size_t cap = 0;

  void reserve(size_t n);
  ~PinnedBuffer() { cudaFreeHost(data); }
};

template <typename T>
void DeviceBuffer<T>::reserve(size_t n) {
  if (n <= cap) return;
  cudaFree(data);
  cuda_check(cudaMalloc(&data, sizeof(T) * n), "malloc batch buffer");
  cap = n;
}

template <typename T>
void DeviceBuffer<T>::upload(const T* v, size_t n) {
  reserve(n == 0 ? 1 : n);
  if (n) cuda_check(cudaMemcpy(data, v, sizeof(T) * n, cudaMemcpyHostToDevice), "batch upload");
}

template <typename T>
void PinnedBuffer<T>::reserve(size_t n) {
  if (n <= cap) return;
  cudaFreeHost(data);
  cuda_check(cudaMallocHost(&data, sizeof(T) * n), "pinned malloc");
  cap = n;
}

}  // namespace scribblez::gpu
