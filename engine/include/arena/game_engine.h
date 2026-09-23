#pragma once

// Owns one agent pair per thread and plays single games on them, handing each
// finished log to a GameSink. The shared core of GameRunner and
// StreamingGameProducer, which differ only in their driving loop and their
// sink. The agents are whatever the --player specs name, so this drives bot
// matches and human games as well as self-play.

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
    // Honor agents' projected_remaining_moves (see MoveDecision), playing out
    // a proven endgame directly instead of prompting turn by turn. Self-play
    // generation turns it on: its compute is better spent on undecided games,
    // and its logs tolerate proof-line moves standing in for agent moves.
    bool respect_projections = false;
    // Play face-up-leaves Scrabble, in which each player's retained tiles are
    // public until they move again (docs/roadmap.md).
    bool face_up_leaves = false;
  };

  // Builds `params.threads` agent pairs, downgrading to 1 (with a warning) if
  // any agent does not support parallelism. Throws util::CleanException on
  // bad params.
  GameEngine(const Params& params, const PlayerFactory::Params& player_params);

  // Agent pairs actually built, which may be fewer than params.threads.
  int num_threads() const { return agents_.size(); }

  // Game g is played with seed() + g.
  uint64_t seed() const { return params_.seed; }

  // The two persistent players' names. Seat-independent, since seats alternate
  // between games.
  std::array<std::string, 2> player_names() const;

  // Play one game on `thread_idx`'s agent pair, with player seats[s] at seat s,
  // and hand the finished log to `sink`. The returned end-of-game actions
  // (PLAY_AGAIN/QUIT) matter only to a serial driver.
  std::pair<EndGameAction, EndGameAction> play(int thread_idx, const std::array<int, 2>& seats,
                                               uint64_t game_idx, GameSink& sink);

 private:
  Params params_;
  std::vector<PlayerFactory::Players> agents_;
};

}  // namespace scribblez
