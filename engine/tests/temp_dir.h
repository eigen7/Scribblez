#pragma once

// A fresh, uniquely named directory under the system temp dir. ctest runs
// every test case as its own process and runs them in parallel, so a fixture
// that used a fixed name would share it across cases, and one case's
// TearDown would delete it under another.

#include "util/exception.h"

#include <filesystem>
#include <stdlib.h>
#include <string>

namespace scribblez::testing {

inline std::filesystem::path make_temp_dir(const std::string& prefix) {
  std::string templ = (std::filesystem::temp_directory_path() / (prefix + "_XXXXXX")).string();
  if (::mkdtemp(templ.data()) == nullptr) throw util::Exception("mkdtemp failed for {}", templ);
  return templ;
}

}  // namespace scribblez::testing
