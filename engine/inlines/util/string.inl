#include "util/string.h"

#include <format>

namespace scribblez::util {

inline std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(delim, start);
    if (end == std::string::npos) end = s.size();
    if (end > start) out.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

inline bool natural_less(const std::string& a, const std::string& b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit(uint8_t(a[i])) && std::isdigit(uint8_t(b[j]))) {
      // Compare maximal digit runs numerically: after skipping leading zeros, the
      // longer run is larger; equal lengths compare digit by digit.
      while (i < a.size() && a[i] == '0') ++i;
      while (j < b.size() && b[j] == '0') ++j;
      size_t ia = i, jb = j;
      while (ia < a.size() && std::isdigit(uint8_t(a[ia]))) ++ia;
      while (jb < b.size() && std::isdigit(uint8_t(b[jb]))) ++jb;
      const size_t la = ia - i, lb = jb - j;
      if (la != lb) return la < lb;
      const int cmp = a.compare(i, la, b, j, lb);
      if (cmp != 0) return cmp < 0;
      i = ia;
      j = jb;
    } else if (a[i] != b[j]) {
      return uint8_t(a[i]) < uint8_t(b[j]);
    } else {
      ++i;
      ++j;
    }
  }
  return i >= a.size() && j < b.size();  // a exhausted first -> the shorter, smaller string
}

inline std::string& to_lower(std::string& s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

inline std::string fmt_dur(double secs) {
  long s = secs + 0.5;
  const long h = s / 3600;
  s %= 3600;
  const long m = s / 60;
  s %= 60;
  if (h > 0) return std::format("{}h{}m", h, m);
  if (m > 0) return std::format("{}m{}s", m, s);
  return std::format("{}s", s);
}

inline bool path_natural_less(const std::filesystem::path& a, const std::filesystem::path& b) {
  return natural_less(a.filename().string(), b.filename().string());
}

}  // namespace scribblez::util
