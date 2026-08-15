#include "nn/trt_util.h"

#include "nn/cuda_util.h"
#include "util/exception.h"

#include <NvInferVersion.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <unistd.h>

namespace scribblez {
namespace nn {

Precision parse_precision(const std::string& s) {
  std::string up = s;
  std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return std::toupper(c); });
  if (up == "FP32") return Precision::kFP32;
  if (up == "FP16") return Precision::kFP16;
  throw util::CleanException("Invalid precision '{}'. Valid values are: FP32, FP16", s);
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
  return std::format("{}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
}

std::string content_hash(const std::vector<char>& bytes) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
  for (char c : bytes) {
    h ^= static_cast<uint8_t>(c);
    h *= 1099511628211ULL;  // FNV-1a 64-bit prime
  }
  return std::format("{:016x}", h);
}

std::string engine_plan_cache_path(const std::string& model_key, Precision precision,
                                   const std::string& profile_tag, bool fast_build,
                                   const std::string& mount_root) {
  const std::string build_tag = fast_build ? "/build_fast" : "";
  return mount_root + "/TensorRT-cache/sm_" + sm_tag() + "/trt_" + trt_version_tag() + build_tag +
         "/fp_" + precision_to_string(precision) + "/" + profile_tag + "/" + model_key + ".engine";
}

std::vector<char> read_file_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::CleanException("Failed to open file: {}", path);
  std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  f.read(bytes.data(), size);
  return bytes;
}

void write_file_bytes(const std::string& path, const char* bytes, size_t size) {
  std::filesystem::path p(path);
  std::filesystem::create_directories(p.parent_path());
  std::string tmp = std::format("{}.tmp.{}.{}", path, ::getpid(), std::random_device{}());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw util::Exception("Failed to open temp file: {}", tmp);
    f.write(bytes, static_cast<std::streamsize>(size));
  }
  std::filesystem::rename(tmp, p);
}

}  // namespace nn
}  // namespace scribblez
