#pragma once

// Real positions for tests: the decision point reached after some HastyBot vs
// HastyBot plies of a seeded game.

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <cstdint>
#include <functional>
#include <optional>

namespace scribblez::testing {

// A decision point: the board, the mover's rack, and the scores from the
// mover's point of view.
struct Position {
  Board board;
  Rack rack;
  int my_score = 0;
  int opp_score = 0;
  int bag_size = 0;

  MoveRequest request(const Dictionary& dict) const {
    static const Rack kHidden;
    return MoveRequest{board, dict, rack, kHidden, my_score, opp_score, bag_size};
  }
};

// The position after `plies` HastyBot-vs-HastyBot moves of game `seed`, or
// nullopt if the game did not stop there (it ended, or the bag was already
// empty).
inline std::optional<Position> hasty_position(const Dictionary& dict, uint64_t seed, int plies) {
  HastyBot a0({.thread_id = 0, .name = "A"}), a1({.thread_id = 0, .name = "B"});
  Game g(a0, a1, dict, seed);
  g.set_max_plies(plies);
  g.play();
  if (!g.truncated()) return std::nullopt;
  const int mover = plies % 2;
  return Position{g.board(), g.rack(mover), g.score(mover), g.score(1 - mover), g.bag_size()};
}

// The first position of game `seed` whose bag size satisfies `want`.
inline std::optional<Position> find_position(const Dictionary& dict, uint64_t seed,
                                             const std::function<bool(int)>& want) {
  for (int plies = 1; plies < 40; ++plies) {
    const std::optional<Position> p = hasty_position(dict, seed, plies);
    if (p && want(p->bag_size)) return p;
  }
  return std::nullopt;
}

}  // namespace scribblez::testing
