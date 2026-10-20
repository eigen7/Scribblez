#pragma once

// Forgone plays: a lexical input for the leave inference model
// (docs/plans/deep_bot_v0.md, "Forgone plays"). For a play and each tile kind
// tau, how much more the best play needing tau would have scored, had the
// player also held tau. A large margin is evidence that the leave lacks tau.
//
// The best play "needing tau" places the played tiles' sub-multiset plus one
// more tau than they hold, so a play the played tiles alone could make never
// counts. Its score is a lower bound on what holding tau offered, since the
// leave's other tiles are unknown. Everything it reads is public: the board
// before the play, the play's tiles and its score.

#include "game/board.h"
#include "game/move.h"
#include "game/tile.h"

#include <array>
#include <cstdint>

namespace scribblez {
class Dictionary;
}

namespace scribblez::belief {

struct ForgonePlays {
  // No margin: the move is not a play of 1..6 tiles, or the board after it
  // holds every copy of tau, so nobody could think tau is in the leave. Which
  // tau get one depends on the board alone, never on a rack: excluding the
  // tiles the mover can see would drop exactly the one-copy tiles they kept.
  static constexpr int8_t kNone = INT8_MIN;
  // Margins clamp to [kFloor, kCeiling]. kFloor also stands for "no play
  // needing tau scores within kFloor of the move".
  static constexpr int kFloor = -50;
  static constexpr int kCeiling = 127;

  std::array<int8_t, TILE_KINDS> margin;

  static ForgonePlays none();
};

// The forgone plays of `move`, played on `before` (whose movegen caches it
// builds from `dict` if needed).
ForgonePlays compute_forgone_plays(const Board& before, const Dictionary& dict, const Move& move);

// The same result by brute force, one exhaustive generation per tau. The
// specification compute_forgone_plays is tested against.
ForgonePlays compute_forgone_plays_reference(const Board& before, const Dictionary& dict,
                                             const Move& move);

}  // namespace scribblez::belief
