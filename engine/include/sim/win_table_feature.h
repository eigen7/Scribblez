#pragma once

// The win table's input (docs/plans/deep_bot_v0.md, "The win table"): what a
// sim leaf reads off a position, and what the table is fitted against. The
// fit and the lookup both go through win_table_feature, so they cannot
// disagree on what a cell means.

#include "game/rack.h"

#include <cstdint>
#include <vector>

namespace scribblez {

// A WinPctTable cell, from the side to move's point of view. It and
// WinTableSample cross the FFI as packed records whose layout format_layout
// serves, so a field added here reaches Python with no signature change.
struct WinTableFeature {
  // The spread plus the static value of the side to move's own leave, rounded
  // to the nearest point.
  int32_t spread;
  // The bag plus the opponent's rack.
  int32_t tiles_unseen;
};
static_assert(sizeof(WinTableFeature) == 8, "WinTableFeature must have no padding");

// `own_leave` is what the side to move kept at their last move. Reads leave
// values from HastyEquity, which must be initialized.
WinTableFeature win_table_feature(int spread, const Rack& own_leave, int tiles_unseen);

// One fitting sample: a position's feature and how the game ended for its
// side to move.
struct WinTableSample {
  WinTableFeature feature;
  float outcome;  // 1 a win, 0.5 a tie, 0 a loss
};
static_assert(sizeof(WinTableSample) == 12, "WinTableSample must have no padding");

// Appends a sample for every fittable pre-move position of every game in a
// loaded .slog buffer, whose magic and version the caller has checked. A
// position is fittable when its side to move has moved before (so has a
// leave), and only agent play follows it (no random-opening ply at or after
// it). Positions past the point where the bag empties are included, since sim
// leaves land there too.
void append_win_table_samples(const char* slog_buf, std::vector<WinTableSample>& out);

}  // namespace scribblez
