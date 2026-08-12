#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// TensorRT helpers that do not themselves need the NvInfer headers, so the
// agent and service code can reason about precision without <NvInfer.h>.

namespace scribblez {
namespace nn {

enum class Precision : uint8_t { kFP32, kFP16 };

// Case-insensitive; throws std::runtime_error on anything else.
Precision parse_precision(const std::string& s);
const char* precision_to_string(Precision precision);

// "10.11.0" -- the linked TensorRT version. A serialized plan is only loadable
// by the same major version that built it.
std::string trt_version_tag();

// Hex-encoded FNV-1a: an exact-content fingerprint of a model file, for uses
// where weights matter (e.g. tagging eval targets with the model behind them).
std::string content_hash(const std::vector<char>& bytes);

// Under <mount>/TensorRT-cache/. Keying on the architecture signature the
// exporter stamps into the ONNX metadata_props, rather than on file contents,
// lets every checkpoint of one architecture share a plan, which the loader
// refits with the checkpoint's own weights (see NeuralNet::load). The path also
// encodes the GPU's compute capability and the TensorRT version, since a plan
// is invalid across either, and `fast_build` plans live in a separate subtree
// so one can never satisfy a full-optimization load.
//
// `profile_tag` names the optimization profile the plan was built for, and is a
// caller's string rather than a number because the two model families size
// different axes: the position net bounds a row batch ("batch_256"), the move
// set net a candidate count ("moves_4096"). A plan is only valid within the
// bounds it was built with, so the tag has to separate them.
std::string engine_plan_cache_path(const std::string& architecture_signature, Precision precision,
                                   const std::string& profile_tag, bool fast_build,
                                   const std::string& mount_root);

// Whole-file read, for the ONNX models and cached plans the loaders consume.
// Throws if the file cannot be opened.
std::vector<char> read_file_bytes(const std::string& path);

// Write `bytes` to `path` atomically (temp file + rename), creating parents.
// The temp name carries the pid and a random suffix, so two processes building
// the same plan concurrently -- self-play workers sharing one cache directory
// -- cannot corrupt each other's rename.
void write_file_bytes(const std::string& path, const char* bytes, size_t size);

}  // namespace nn
}  // namespace scribblez
