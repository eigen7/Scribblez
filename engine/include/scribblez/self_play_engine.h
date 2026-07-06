#pragma once

// SelfPlayEngine owns the per-thread agent pairs and the self-play primitive:
// build the agents once, then play individual games on a chosen thread,
// routing each finished game's log to a GameSink. It is the shared core under
// both GameRunner (disk output, fixed game count, PLAY_AGAIN/QUIT loop) and the
// streaming producer (ring-buffer output, unbounded loop), so the two differ
// only in their driving loop and their sink.

#include "scribblez/agent.h"
#include "scribblez/game_sink.h"
#include "scribblez/player_factory.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

class SelfPlayEngine {
 public:
  struct Params {
    int threads = 1;       // requested number of parallel game threads
    uint64_t seed = 0;     // base seed; game g is played with seed + g
    int handicap_max = 0;  // if > 0, gift a random player [0, max] head-start points
    // If > 0, each game opens with K uniformly-random plies (Game's
    // random-opening mode), K drawn per game as an exponential with this mean,
    // rounded to the nearest integer.
    double random_opening_mean = 0.0;
  };

  // Builds `params.threads` agent pairs (downgrading to 1, with a warning, if
  // any agent does not support parallelism). Throws scribblez::Exception on bad
  // params (threads < 1).
  SelfPlayEngine(const Params& params, const PlayerFactory::Params& player_params);

  // Number of agent pairs actually built (== usable parallel threads).
  int num_threads() const { return static_cast<int>(agents_.size()); }

  // Base seed; game g is played with seed() + g.
  uint64_t seed() const { return params_.seed; }

  // Display names of the two persistent players (seat-independent identities).
  std::array<std::string, 2> player_names() const;

  // Play one game on `thread_idx`'s agent pair. `seats[s]` is the player index
  // seated at seat s. Constructs the Game with seed() + game_idx, applies the
  // handicap, plays, and hands the finished log storage to `sink`. Returns each
  // seat's end-of-game action (only meaningful on the serial path).
  std::pair<EndGameAction, EndGameAction> play(int thread_idx, const std::array<int, 2>& seats,
                                               uint64_t game_idx, GameSink& sink);

 private:
  Params params_;
  std::vector<PlayerFactory::Players> agents_;
};

}  // namespace scribblez
