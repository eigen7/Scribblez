#pragma once

#include <cstddef>

// Thin, throwing wrappers around the handful of CUDA runtime calls the
// TensorRT inference path needs. Every call checks the returned cudaError_t and
// throws scribblez::Exception on failure, so callers never have to inspect
// status codes. The CUDA headers are confined to cuda_util.cpp; consumers see
// only opaque void* device pointers and the cudaStream_t typedef below.

// cudaStream_t is `struct CUstream_st*`; forward-declare it so this header does
// not pull in <cuda_runtime_api.h>.
struct CUstream_st;

namespace scribblez {
namespace nn {

using stream_t = CUstream_st*;

// "8.9" for an RTX 4090. Used to key the engine-plan cache, since a plan is
// only valid for the compute capability it was built on.
const char* sm_tag();

void set_device(int device_id);

stream_t create_stream();
void destroy_stream(stream_t stream);
void synchronize_stream(stream_t stream);

void* device_malloc(size_t n_bytes);
void device_free(void* ptr);

// Pinned host allocation, so host<->device copies can run asynchronously on the
// stream. Freed with host_free().
void* host_malloc(size_t n_bytes);
void host_free(void* ptr);

void host_to_device_async(stream_t stream, void* dst, const void* src, size_t n_bytes);
void device_to_host_async(stream_t stream, void* dst, const void* src, size_t n_bytes);

}  // namespace nn
}  // namespace scribblez
