#pragma once

#include <cstdint>
#include <string>
#include <vector>

// TensorRT-related helpers that do not themselves need the NvInfer headers:
// precision selection, an FNV-1a content hash for fingerprinting model files,
// and the on-disk engine-plan cache-path layout. Keeping these out of
// neural_net.h lets the agent/service code reason about precision without
// including <NvInfer.h>.

namespace scribblez {
namespace nn {

enum class Precision : uint8_t { kFP32, kFP16 };

// Parse "FP32"/"FP16" (case-insensitive); throws std::runtime_error otherwise.
Precision parse_precision(const std::string& s);
const char* precision_to_string(Precision precision);

// "10.11.0" -- the linked TensorRT library version. A serialized engine plan is
// only loadable by the same major TensorRT version that built it.
std::string trt_version_tag();

// 64-bit FNV-1a hash of `bytes`, hex-encoded. Used as an exact-content
// fingerprint of a model file where weights matter (e.g. tagging generated
// eval targets with the model that produced them).
std::string content_hash(const std::vector<char>& bytes);

// Absolute path of the cached engine plan for a model with the given
// architecture signature (the "model-architecture-signature" entry the
// exporter stamps into the ONNX metadata_props), built at `precision` for
// `batch_size`. Keying on the architecture rather than the file contents lets
// every checkpoint of one architecture share a single cached plan; the loader
// refits the plan with the checkpoint's own weights (see NeuralNet::load).
// The path also encodes the GPU's compute capability and the TensorRT
// version, since a plan is invalid across either. Lives under
// <mount>/TensorRT-cache/.
//
// `fast_build` engines (TensorRT builder optimization level 0: minimal build
// effort, much lower inference throughput) are cached under a separate subtree
// so a fast-built plan can never satisfy a normal full-optimization load.
std::string engine_plan_cache_path(const std::string& architecture_signature, Precision precision,
                                   int batch_size, bool fast_build, const std::string& mount_root);

}  // namespace nn
}  // namespace scribblez
