#pragma once

#include <cstddef>

// Thin wrappers around the CUDA runtime calls the TensorRT inference path
// needs, each throwing util::Exception on failure so callers never inspect
// status codes. The CUDA headers stay confined to cuda_util.cpp; consumers see
// only opaque void* device pointers and the stream typedef below.

// cudaStream_t is `struct CUstream_st*`.
struct CUstream_st;

namespace scribblez {
namespace nn {

using stream_t = CUstream_st*;

// "8.9" for an RTX 4090. Keys the engine-plan cache, a plan being valid only
// for the compute capability it was built on.
const char* sm_tag();

void set_device(int device_id);

stream_t create_stream();
void destroy_stream(stream_t stream);
void synchronize_stream(stream_t stream);

void* device_malloc(size_t n_bytes);
void device_free(void* ptr);

// Pinned, so host<->device copies can run asynchronously on the stream.
void* host_malloc(size_t n_bytes);
void host_free(void* ptr);

void host_to_device_async(stream_t stream, void* dst, const void* src, size_t n_bytes);
void device_to_host_async(stream_t stream, void* dst, const void* src, size_t n_bytes);

}  // namespace nn
}  // namespace scribblez
