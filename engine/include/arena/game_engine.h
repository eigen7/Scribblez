#pragma once

// Owns the per-thread agent pairs and the play-one-game primitive: build the
// agents once, then play individual games on a chosen thread, routing each
// finished log to a GameSink. The shared core under both GameRunner and the
// streaming producer, which differ only in their driving loop and their sink.
// The two agents are whatever the --player specs named, so this drives a bot
// match or a human game as readily as self-play generation.

#include "agent/agent.h"
#include "agent/player_factory.h"
#include "arena/game_sink.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

class GameEngine {
 public:
  struct Params {
    int threads = 1;       // requested number of parallel game threads
    uint64_t seed = 0;     // base seed; game g is played with seed + g
    int handicap_max = 0;  // if > 0, gift a random player [0, max] head-start points
    // If > 0, each game opens with K uniformly-random plies, K drawn per game
    // as an exponential with this mean, rounded to the nearest integer.
    double random_opening_mean = 0.0;
    // Respect agents' projected_remaining_moves annotations (see MoveDecision),
    // fast-tracking a proven endgame instead of prompting turn by turn. Off by
    // default; self-play generation is the special case that turns it on, its
    // compute belonging to undecided games and its logs tolerating proof-line
    // stand-ins for agent moves.
    bool respect_projections = false;
    // Play face-up-leaves Scrabble, in which each player's retained tiles are
    // public until they move again (docs/roadmap.md).
    bool face_up_leaves = false;
  };

  // Builds `params.threads` agent pairs, downgrading to 1 (with a warning) if
  // any agent does not support parallelism. Throws util::CleanException on
  // bad params.
  GameEngine(const Params& params, const PlayerFactory::Params& player_params);

  // Agent pairs actually built, which may be below params.threads; see the
  // constructor.
  int num_threads() const { return static_cast<int>(agents_.size()); }

  // Game g is played with seed() + g.
  uint64_t seed() const { return params_.seed; }

  // The two persistent players' names. Seat-independent, since seats alternate
  // between games.
  std::array<std::string, 2> player_names() const;

  // Play one game on `thread_idx`'s agent pair, `seats[s]` being the player
  // index at seat s, and hand the finished log storage to `sink`. The returned
  // end-of-game actions are meaningful only on the serial path.
  std::pair<EndGameAction, EndGameAction> play(int thread_idx, const std::array<int, 2>& seats,
                                               uint64_t game_idx, GameSink& sink);

 private:
  Params params_;
  std::vector<PlayerFactory::Players> agents_;
};

}  // namespace scribblez
