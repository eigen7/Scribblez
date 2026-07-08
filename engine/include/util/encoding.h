#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace scribblez::util {

// Standard Base64 (RFC 4648) encoding of len bytes, with '=' padding.
std::string base64(const uint8_t* data, size_t len);

}  // namespace scribblez::util
