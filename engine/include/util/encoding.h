#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace util {

// Standard Base64 (RFC 4648) encoding of len bytes, with '=' padding.
std::string base64(const uint8_t* data, size_t len);

// Decode standard Base64 to raw bytes. Whitespace and any non-alphabet
// characters are skipped; decoding stops at the first '=' padding.
std::string base64_decode(const std::string& in);

}  // namespace util
