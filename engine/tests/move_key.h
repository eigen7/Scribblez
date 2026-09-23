#pragma once

// Order-free identity for generated moves, for tests that compare move lists
// or check a move's membership in one.

#include "game/move.h"

#include <algorithm>
#include <compare>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace scribblez::testing {

// One placed tile of a play, ordered by square.
struct KeyedPlacement {
  int r, c, code;
  auto operator<=>(const KeyedPlacement&) const = default;
};

// The tiles a play places, sorted by square; empty for a non-play.
inline std::vector<KeyedPlacement> sorted_placements(const Move& m) {
  std::vector<KeyedPlacement> tiles;
  if (m.type() != MoveType::PLAY) return tiles;
  const bool horiz = m.horizontal();
  uint16_t mask = m.square_mask();
  int gi = 0;
  for (int pos = 0; mask; ++pos, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horiz ? m.start() : pos;
    const int c = horiz ? pos : m.start();
    tiles.push_back({r, c, m.glyph(gi++).code()});
  }
  std::sort(tiles.begin(), tiles.end());
  return tiles;
}

// A canonical key for a play: its placed tiles (by square) plus its score. The
// key omits orientation, so a single-tile play generated from either direction
// compares equal.
inline std::string move_key(const Move& m) {
  std::string k;
  char buf[32];
  for (const KeyedPlacement& t : sorted_placements(m)) {
    std::snprintf(buf, sizeof(buf), "%d,%d,%d;", t.r, t.c, t.code);
    k += buf;
  }
  std::snprintf(buf, sizeof(buf), "|%d", m.score());
  k += buf;
  return k;
}

inline std::set<std::string> key_set(const std::vector<Move>& ms) {
  std::set<std::string> s;
  for (const Move& m : ms) s.insert(move_key(m));
  return s;
}

}  // namespace scribblez::testing
