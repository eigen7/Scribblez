#pragma once

// Build a new .slog from a selection of games drawn out of existing ones. The
// games are copied verbatim, only the file header and per-game start offsets
// being recomputed, so the result is a standard .slog -- loadable by all the
// same tooling -- holding a curated subset (e.g. a frozen evaluation set
// sampled from the held-out test split).

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

// One selected game: a source .slog path and the game index within it.
struct SlogPick {
  std::string path;
  int64_t game_idx;
};

// False on any I/O error, bad header, or out-of-range index.
bool write_slog_subset(const std::string& dst_path, const std::vector<SlogPick>& picks);

}  // namespace binlog
}  // namespace scribblez
