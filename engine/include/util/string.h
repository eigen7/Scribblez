#pragma once

#include <sstream>
#include <string>

namespace util {

// Lowercase the ASCII letters A-Z in place; other bytes are left untouched.
// Returns the same string for call chaining.
inline std::string& to_lower(std::string& s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

// Compact human-readable duration: "6h32m", "45m12s", or "30s".
inline std::string fmt_dur(double secs) {
  long s = static_cast<long>(secs + 0.5);
  const long h = s / 3600;
  s %= 3600;
  const long m = s / 60;
  s %= 60;
  std::ostringstream o;
  if (h > 0)
    o << h << "h" << m << "m";
  else if (m > 0)
    o << m << "m" << s << "s";
  else
    o << s << "s";
  return o.str();
}

}  // namespace util
