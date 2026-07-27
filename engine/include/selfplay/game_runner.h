#pragma once

#include "agent/player_factory.h"
#include "game/game.h"
#include "selfplay/game_sink.h"
#include "selfplay/self_play_engine.h"
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

// Drives a fixed series of self-play games to disk. Owns a SelfPlayEngine (the
// agents plus the per-game primitive) and the win/loss tally, and is itself the
// GameSink. It alternates seats each game and honors each agent's EndGameResult
// to extend (PLAY_AGAIN) or shorten (QUIT) the series past `--games`.
class GameRunner : public GameSink {
 public:
  struct Params {
    int games = 1;               // minimum number of games to play
    std::string log_dir;         // if non-empty, one <id>.gcg per game
    std::string binary_log_dir;  // if non-empty, batched .slog files
    int games_per_file = 100;    // only used with binary_log_dir
    // SelfPlayEngine downgrades to 1 thread, with a warning, when a player does
    // not support parallelism.
    int threads = util::default_thread_count();
    int random_handicap_max = 0;       // if > 0, gift a random player a head-start of
                                       // P points, P uniform in [0, this], each game
    double random_opening_mean = 0.0;  // if > 0, open each game with K uniformly-
                                       // random plies, K ~ round(Exp(this mean))
    bool respect_projections = false;  // fast-track games whose end an agent proved;
                                       // self-play generation turns this on
    int progress_secs = 10;            // games-done/rate/ETA line interval; 0 disables,
                                       // and only the parallel batch loop prints one
    bool verbose = false;              // per-game + batch summaries to stderr

    void add_options(boost::program_options::options_description& desc);
  };

  // Validates the params, builds one agent pair per thread, and loads the
  // lexicon. Takes its starting seed from SeedProducer::instance(), which the
  // caller must already have reseeded if reproducibility is wanted. Throws
  // scribblez::Exception on a user-visible error, having explained it to
  // stderr.
  GameRunner(const Params& runner_params, const PlayerFactory::Params& player_params);
  ~GameRunner();

  // On one thread the loop honors PLAY_AGAIN/QUIT; on several it plays exactly
  // params.games games in parallel.
  void run();

  // Prints a user-facing setup hint and throws Exception on failure.
  static const Dictionary& load_dictionary_or_throw();

  // Writes the finished game as an optional .gcg, tallies it, and appends it to
  // the .slog writer. Called from every game thread; thread-safe.
  void on_game(GameLogStorage&& log, const std::array<int, 2>& seats) override;

 private:
  // Tally indexed by *player identity* rather than seat, seats alternating
  // every game. Thread-safe.
  class Results;

  // Prints a progress line every progress_secs until `done` is set, polling it
  // at 10 Hz so it exits promptly once the workers finish.
  void run_progress_monitor(const std::atomic<bool>& done, std::chrono::steady_clock::time_point t0,
                            uint64_t total) const;

  Params params_;
  uint64_t seed_;
  SelfPlayEngine engine_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<binlog::BinaryLogWriter> binary_writer_;
};

}  // namespace scribblez
