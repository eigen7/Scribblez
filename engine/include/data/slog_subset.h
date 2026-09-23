#pragma once

// Builds a new .slog from games selected out of existing ones, e.g. a frozen
// evaluation set sampled from the held-out test split. Games are copied
// verbatim; only the file header and start offsets are recomputed.

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

struct SlogPick {
  std::string path;
  int64_t game_idx;
};

// Returns false on any I/O error, bad header, or out-of-range index.
bool write_slog_subset(const std::string& dst_path, const std::vector<SlogPick>& picks);

}  // namespace binlog
}  // namespace scribblez
