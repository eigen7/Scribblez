#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// TensorRT helpers that do not need the NvInfer headers, so agent and service
// code can handle precision settings and plan-cache paths without them.

namespace scribblez {
namespace nn {

enum class Precision : uint8_t { kFP32, kFP16, kBF16 };

// Case-insensitive; throws util::CleanException on anything else.
Precision parse_precision(const std::string& s);
const char* precision_to_string(Precision precision);

// The linked TensorRT version, e.g. "10.11.0". Part of the plan-cache path,
// since a serialized plan is loadable only by the version that built it.
std::string trt_version_tag();

// A hex FNV-1a fingerprint of a model file's exact bytes, for recording which
// weights produced some output. Unlike the architecture signature, it tells
// checkpoints apart.
std::string content_hash(const std::vector<char>& bytes);

// Where a plan is cached, under <mount_root>/TensorRT-cache/. The path
// encodes everything a plan is only valid for:
//   - the GPU's compute capability and the TensorRT version;
//   - `fast_build`, so a quick plan never satisfies a full-optimization load;
//   - the precision;
//   - `profile_tag`, the dynamic axis and its bound, e.g. "batch_256" or
//     "moves_4096" (Spec::kAxisTag plus max_rows). A string because the
//     families bound different axes;
//   - `model_key`, which decides which models share a plan. NeuralNetBase
//     passes the architecture signature, so every checkpoint of one
//     architecture shares a plan and is refitted into it on load.
std::string engine_plan_cache_path(const std::string& model_key, Precision precision,
                                   const std::string& profile_tag, bool fast_build,
                                   const std::string& mount_root);

// Throws if the file cannot be opened.
std::vector<char> read_file_bytes(const std::string& path);

// Write `bytes` to `path` atomically (temp file + rename), creating parent
// directories. The temp name carries the pid and a random suffix, so processes
// that build the same plan concurrently into a shared cache directory cannot
// clobber each other's temp file.
void write_file_bytes(const std::string& path, const char* bytes, size_t size);

}  // namespace nn
}  // namespace scribblez
