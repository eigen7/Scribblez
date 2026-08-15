#pragma once

#include "agent/player_factory.h"
#include "arena/game_engine.h"
#include "arena/game_sink.h"
#include "game/game_log.h"
#include "util/misc.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

namespace binlog {
class BinaryLogWriter;
}

class Dictionary;

// Drives a fixed series of games between two agents to disk -- a bot-vs-bot
// match, a human game, or self-play generation, according to the --player
// specs. Owns a GameEngine (the agents plus the per-game primitive) and the
// win/loss tally, and is itself the GameSink. It alternates seats each game and
// honors each agent's EndGameResult to extend (PLAY_AGAIN) or shorten (QUIT)
// the series past `--games`.
class GameRunner : public GameSink {
 public:
  struct Params {
    int games = 1;               // minimum number of games to play
    std::string log_dir;         // if non-empty, one <id>.gcg per game
    std::string binary_log_dir;  // if non-empty, batched .slog files
    // GameEngine downgrades to 1 thread, with a warning, when a player does
    // not support parallelism.
    int threads = util::default_thread_count();
    int random_handicap_max = 0;       // if > 0, gift a random player a head-start of
                                       // P points, P uniform in [0, this], each game
    double random_opening_mean = 0.0;  // if > 0, open each game with K uniformly-
                                       // random plies, K ~ round(Exp(this mean))
    bool respect_projections = false;  // fast-track games whose end an agent proved;
                                       // self-play generation turns this on
    bool face_up_leaves = false;       // play the face-up-leaves variant, in which each
                                       // player's retained tiles are public
    bool paired = false;               // games 2k and 2k+1 share one game seed with the
                                       // seats swapped, so per-seed tile luck cancels
                                       // out of paired comparisons; needs an even --games
    std::string results_file;          // if non-empty, one JSON line per finished game
    int progress_secs = 10;            // games-done/rate/ETA line interval; 0 disables,
                                       // and only the parallel batch loop prints one
    bool verbose = false;              // per-game + batch summaries to stderr

    void add_options(boost::program_options::options_description& desc);
  };

  // Validates the params, builds one agent pair per thread, and loads the
  // lexicon. Takes its starting seed from SeedProducer::instance(), which the
  // caller must already have reseeded if reproducibility is wanted. Throws
  // util::CleanException on a user-visible error.
  GameRunner(const Params& runner_params, const PlayerFactory::Params& player_params);
  ~GameRunner();

  // On one thread the loop honors PLAY_AGAIN/QUIT; on several it plays exactly
  // params.games games in parallel.
  void run();

  // Writes the finished game as an optional .gcg, tallies it, and appends it to
  // the .slog writer. Called from every game thread; thread-safe.
  void on_game(GameLogStorage&& log, const std::array<int, 2>& seats) override;

 private:
  // Tally indexed by *player identity* rather than seat, seats alternating
  // every game, plus the optional per-game JSON results stream. Thread-safe.
  class Results;

  // The seed index game `game_idx` is played with: the games of a pair share
  // one seed in paired mode.
  uint64_t seed_index(uint64_t game_idx) const { return params_.paired ? game_idx / 2 : game_idx; }

  // Prints a progress line every progress_secs until `done` is set, polling it
  // at 10 Hz so it exits promptly once the workers finish.
  void run_progress_monitor(const std::atomic<bool>& done, std::chrono::steady_clock::time_point t0,
                            uint64_t total) const;

  Params params_;
  uint64_t seed_;
  GameEngine engine_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<binlog::BinaryLogWriter> binary_writer_;
};

}  // namespace scribblez
