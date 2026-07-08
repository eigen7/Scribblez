#pragma once

#include <cstdint>
#include <string>

namespace scribblez::util {

// Compute the SHA-1 digest of msg, writing the 20-byte result to out.
void sha1(const std::string& msg, uint8_t out[20]);

}  // namespace scribblez::util
