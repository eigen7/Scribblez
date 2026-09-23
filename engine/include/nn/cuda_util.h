#pragma once

#include <cstddef>

// Thin wrappers around the CUDA runtime calls the TensorRT inference path
// needs. Each throws util::Exception on failure, so callers never inspect
// status codes. The CUDA headers stay confined to cuda_util.cpp: consumers see
// only opaque void* device pointers and the stream typedef below.

// cudaStream_t is `struct CUstream_st*`.
struct CUstream_st;

namespace scribblez {
namespace nn {

using stream_t = CUstream_st*;

// The current device's compute capability, e.g. "8.9" for an RTX 4090. Keys
// the engine-plan cache, since a plan is valid only on the compute capability
// it was built for.
const char* sm_tag();

// Compute-capability major version of the current device (8 for Ampere and
// Ada, 9 for Hopper). BF16 tensor cores require >= 8.
int compute_capability_major();

void set_device(int device_id);

stream_t create_stream();
void destroy_stream(stream_t stream);
void synchronize_stream(stream_t stream);

// Bytes of device memory in use on the current device (total minus free), for
// tools that report what a loaded engine costs.
size_t device_memory_used();

void* device_malloc(size_t n_bytes);
void device_free(void* ptr);

// Page-locked host memory, which cudaMemcpyAsync needs to run asynchronously.
void* host_malloc(size_t n_bytes);
void host_free(void* ptr);

void host_to_device_async(stream_t stream, void* dst, const void* src, size_t n_bytes);
void device_to_host_async(stream_t stream, void* dst, const void* src, size_t n_bytes);

}  // namespace nn
}  // namespace scribblez
