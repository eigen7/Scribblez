#include "util/string.h"

namespace scribblez::util {

// Natural-order ("natural sort") comparison of two strings: like a lexicographic
// compare, except that maximal runs of digits compare by numeric value (ignoring
// leading zeros). So "pos-2" orders before "pos-10", unlike a plain string sort.
inline bool natural_less(const std::string& a, const std::string& b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit(static_cast<unsigned char>(a[i])) &&
        std::isdigit(static_cast<unsigned char>(b[j]))) {
      // Compare the two maximal digit runs by numeric value. Skip leading zeros,
      // then a longer remaining run of significant digits is the larger number; on
      // equal length, fall back to a digit-by-digit compare.
      while (i < a.size() && a[i] == '0') ++i;
      while (j < b.size() && b[j] == '0') ++j;
      size_t ia = i, jb = j;
      while (ia < a.size() && std::isdigit(static_cast<unsigned char>(a[ia]))) ++ia;
      while (jb < b.size() && std::isdigit(static_cast<unsigned char>(b[jb]))) ++jb;
      const size_t la = ia - i, lb = jb - j;
      if (la != lb) return la < lb;
      const int cmp = a.compare(i, la, b, j, lb);
      if (cmp != 0) return cmp < 0;
      i = ia;
      j = jb;
    } else if (a[i] != b[j]) {
      return static_cast<unsigned char>(a[i]) < static_cast<unsigned char>(b[j]);
    } else {
      ++i;
      ++j;
    }
  }
  return i >= a.size() && j < b.size();  // a exhausted first -> the shorter, smaller string
}

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

// Order filesystem paths by their filename in natural-sort order (so "pos-2.gcg"
// precedes "pos-10.gcg"). Suitable as a std::sort comparator.
inline bool path_natural_less(const std::filesystem::path& a, const std::filesystem::path& b) {
  return natural_less(a.filename().string(), b.filename().string());
}

}  // namespace scribblez::util
