#include "util/hash.h"

#include "util/io.h"

#include <cstdint>
#include <format>

namespace scribblez::util {

std::string content_hash(std::span<const char> bytes) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
  for (char c : bytes) {
    h ^= uint8_t(c);
    h *= 1099511628211ULL;  // FNV-1a 64-bit prime
  }
  return std::format("{:016x}", h);
}

std::string file_content_hash(const std::string& path) {
  return content_hash(read_file_bytes(path));
}

}  // namespace scribblez::util
