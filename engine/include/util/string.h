#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace scribblez::util {

// Lexicographic, except that digit runs compare by numeric value: "pos-2"
// orders before "pos-10".
inline bool natural_less(const std::string& a, const std::string& b);

// In place, ASCII only; returns the same string for chaining.
inline std::string& to_lower(std::string& s);

// "6h32m", "45m12s", or "30s".
inline std::string fmt_dur(double secs);

// Drops empty tokens: "a,,b" -> {"a", "b"}.
inline std::vector<std::string> split(const std::string& s, char delim);

// natural_less on the filenames. Suitable as a std::sort comparator.
inline bool path_natural_less(const std::filesystem::path& a, const std::filesystem::path& b);

// Standard Base64 (RFC 4648), with '=' padding.
std::string base64(const uint8_t* data, size_t len);

void sha1(const std::string& msg, uint8_t out[20]);

}  // namespace scribblez::util

#include "inlines/util/string.inl"
