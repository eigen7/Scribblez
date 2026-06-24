#pragma once

#include <string>

namespace util {

// Lowercase the ASCII letters A-Z in place; other bytes are left untouched.
// Returns the same string for call chaining.
inline std::string& to_lower(std::string& s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

}  // namespace util
