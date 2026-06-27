#include "scribblez/nn/trt_util.h"

#include "scribblez/nn/cuda_util.h"

#include <NvInferVersion.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>

namespace scribblez {
namespace nn {

Precision parse_precision(const std::string& s) {
  std::string up = s;
  std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return std::toupper(c); });
  if (up == "FP32") return Precision::kFP32;
  if (up == "FP16") return Precision::kFP16;
  throw std::runtime_error("Invalid precision '" + s + "'. Valid values are: FP32, FP16");
}

const char* precision_to_string(Precision precision) {
  switch (precision) {
    case Precision::kFP32:
      return "FP32";
    case Precision::kFP16:
      return "FP16";
  }
  return "FP32";
}

std::string trt_version_tag() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH);
}

std::string content_hash(const std::vector<char>& bytes) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
  for (char c : bytes) {
    h ^= static_cast<uint8_t>(c);
    h *= 1099511628211ULL;  // FNV-1a 64-bit prime
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

std::string engine_plan_cache_path(const std::string& model_hash, Precision precision,
                                   int batch_size, const std::string& mount_root) {
  return mount_root + "/TensorRT-cache/sm_" + sm_tag() + "/trt_" + trt_version_tag() + "/fp_" +
         precision_to_string(precision) + "/batch_" + std::to_string(batch_size) + "/" +
         model_hash + ".engine";
}

}  // namespace nn
}  // namespace scribblez
