#pragma once

#include "util/string.h"

#include <filesystem>

namespace util {

// Order filesystem paths by their filename in natural-sort order (so "pos-2.gcg"
// precedes "pos-10.gcg"). Suitable as a std::sort comparator.
inline bool path_natural_less(const std::filesystem::path& a, const std::filesystem::path& b) {
  return natural_less(a.filename().string(), b.filename().string());
}

}  // namespace util
