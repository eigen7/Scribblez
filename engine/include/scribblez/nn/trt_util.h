#pragma once

#include <cstdint>
#include <string>
#include <vector>

// TensorRT-related helpers that do not themselves need the NvInfer headers:
// precision selection, the FNV-1a content hash used to key cached engine plans,
// and the on-disk cache-path layout. Keeping these out of neural_net.h lets the
// agent/service code reason about precision without including <NvInfer.h>.

namespace scribblez {
namespace nn {

enum class Precision : uint8_t { kFP32, kFP16 };

// Parse "FP32"/"FP16" (case-insensitive); throws std::runtime_error otherwise.
Precision parse_precision(const std::string& s);
const char* precision_to_string(Precision precision);

// "10.11.0" -- the linked TensorRT library version. A serialized engine plan is
// only loadable by the same major TensorRT version that built it.
std::string trt_version_tag();

// 64-bit FNV-1a hash of `bytes`, hex-encoded. Used as a content fingerprint of
// the ONNX model so structurally identical models reuse a cached plan.
std::string content_hash(const std::vector<char>& bytes);

// Absolute path of the cached engine plan for a model with the given content
// hash, built at `precision` for `batch_size`. The path also encodes the GPU's
// compute capability and the TensorRT version, since a plan is invalid across
// either. Lives under <mount>/TensorRT-cache/.
std::string engine_plan_cache_path(const std::string& model_hash, Precision precision,
                                   int batch_size, const std::string& mount_root);

}  // namespace nn
}  // namespace scribblez
