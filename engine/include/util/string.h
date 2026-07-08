#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace scribblez::util {

// Natural-order ("natural sort") comparison of two strings: like a lexicographic
// compare, except that maximal runs of digits compare by numeric value (ignoring
// leading zeros). So "pos-2" orders before "pos-10", unlike a plain string sort.
inline bool natural_less(const std::string& a, const std::string& b);

// Lowercase the ASCII letters A-Z in place; other bytes are left untouched.
// Returns the same string for call chaining.
inline std::string& to_lower(std::string& s);

// Compact human-readable duration: "6h32m", "45m12s", or "30s".
inline std::string fmt_dur(double secs);

// Order filesystem paths by their filename in natural-sort order (so "pos-2.gcg"
// precedes "pos-10.gcg"). Suitable as a std::sort comparator.
inline bool path_natural_less(const std::filesystem::path& a, const std::filesystem::path& b);

// Standard Base64 (RFC 4648) encoding of len bytes, with '=' padding.
std::string base64(const uint8_t* data, size_t len);

// Compute the SHA-1 digest of msg, writing the 20-byte result to out.
void sha1(const std::string& msg, uint8_t out[20]);

}  // namespace scribblez::util

#include "inlines/util/string.inl"
