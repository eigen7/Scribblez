#pragma once

// Loads the real NWL23 leave values into the HastyEquity singleton, for tests
// whose fixtures (self-play games, expected best moves) depend on them.
//
// Always init(), never ensure_initialized(): the singleton is process-wide, and
// other suites in the same test binary install a synthetic table
// (synthetic_equity.h) that ensure_initialized() would keep. Which suites ran
// first depends on gtest's shard and filter, so a test that leans on
// ensure_initialized() passes alone and fails in ctest.

#include "lexicon/hasty_equity.h"

#include <filesystem>
#include <string>

namespace scribblez::testing {

// False, loading nothing, when the Macondo leaves file is absent.
inline bool install_nwl23_hasty_equity() {
  const std::string leaves = HastyEquity::default_leaves_path("NWL23");
  if (!std::filesystem::exists(leaves)) return false;
  HastyEquity::init(leaves, HastyEquity::default_peg_path());
  return true;
}

}  // namespace scribblez::testing
