#pragma once

// Build a new .slog file from a selection of games drawn out of existing
// .slog files. The selected games are copied verbatim (initial racks + turn
// blobs + their sampled-turn / score metadata) into the destination in the
// given order; only the file header and per-game start offsets are recomputed.
// This produces a standard .slog -- inspectable and loadable by all the same
// tooling as training data -- holding a curated subset of positions (e.g. a
// frozen evaluation set sampled from the held-out test split).

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

// Write the picked games, in order, to a new .slog at `dst_path`. Returns
// true on success; false on any I/O error, bad header, or out-of-range index.
bool write_slog_subset(const std::string& dst_path, const std::vector<SlogPick>& picks);

}  // namespace binlog
}  // namespace scribblez
