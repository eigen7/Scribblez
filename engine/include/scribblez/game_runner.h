#pragma once

#include "scribblez/game.h"
#include "scribblez/game_sink.h"
#include "scribblez/player_factory.h"
#include "scribblez/self_play_engine.h"
#include "util/hardware.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

// Forward-declared so Params::add_options() can register options without
// pulling boost::program_options into every consumer of this header.
namespace boost::program_options {
class options_description;
}

namespace scribblez {

namespace binlog {
class BinaryLogWriter;
}

class Dictionary;

// Drives a fixed series of self-play games to disk. Owns a SelfPlayEngine (the
// agents + per-game primitive) and the win/loss tally, and is itself the
// GameSink: each finished game is written as an optional .gcg, tallied, and
// appended to the binary .slog writer. Alternates seats each game and honors
// each agent's EndGameResult to extend (PLAY_AGAIN) or shorten (QUIT) the
// series past the requested `--games` count. Supports parallel execution via a
// thread pool (--threads / -t); human players disable parallelism because they
// own an interactive browser session.
class GameRunner : public GameSink {
 public:
  struct Params {
    int games = 1;               // minimum number of games to play
    std::string log_dir;         // if non-empty, write one <id>.gcg per game here
    std::string binary_log_dir;  // if non-empty, write batched .slog files here
    int games_per_file = 100;    // games per .slog file (only used with binary_log_dir)
    // Parallel game threads; defaults to every available logical processor.
    // SelfPlayEngine downgrades to 1 (with a warning) when a player does not
    // support parallelism (e.g. a human seat).
    int threads = util::default_thread_count();
    int random_handicap_max = 0;       // if > 0, gift a random player a head-start of
                                       // P points, P uniform in [0, this], each game
    double random_opening_mean = 0.0;  // if > 0, open each game with K uniformly-
                                       // random plies, K ~ round(Exp(this mean))
    int progress_secs = 10;            // print a games-done/rate/ETA line to stderr
                                       // every this many seconds (0 disables); only
                                       // active in the parallel batch loop
    bool verbose = false;              // per-game + batch summaries to stderr

    void add_options(boost::program_options::options_description& desc);
  };

  // Constructs the runner from the two Params structs. Validates the params,
  // builds one agent pair per thread (checking parallelism support and
  // downgrading to 1 thread with a warning if any player cannot run in
  // parallel), and loads the lexicon. Pulls its starting seed from
  // SeedProducer::instance() (which the caller is responsible for having
  // reseeded from --seed if reproducibility is desired). Throws
  // scribblez::Exception on user-visible errors (bad --games/--threads,
  // missing lexicon, bad log dir), having already printed an explanation
  // to stderr.
  GameRunner(const Params& runner_params, const PlayerFactory::Params& player_params);
  ~GameRunner();

  // Run the game loop. For threads==1 the loop honors PLAY_AGAIN/QUIT from
  // agents; for threads>1 it plays exactly params.games games in parallel.
  // Returns normally on a clean end.
  void run();

  // Load and return the active lexicon dictionary, printing a user-facing
  // setup hint and throwing Exception on failure.
  static const Dictionary& load_dictionary_or_throw();

  // GameSink: write the finished game as an optional .gcg, tally it, and append
  // it to the binary .slog writer. Called from every game thread; thread-safe.
  void on_game(GameLogStorage&& log, const std::array<int, 2>& seats) override;

 private:
  // Per-game-and-batch tally, indexed by *player identity* rather than seat
  // (seats alternate every game). Thread-safe via an internal mutex.
  class Results;

  // Body of the monitor thread spawned during the parallel batch loop: every
  // progress_secs seconds, prints a games-done/rate/ETA line to stderr until
  // `done` is set. `t0` is the batch start time and `total` the target game
  // count. Polls `done` at 10 Hz so it exits promptly once the workers finish.
  void run_progress_monitor(const std::atomic<bool>& done, std::chrono::steady_clock::time_point t0,
                            uint64_t total) const;

  Params params_;
  uint64_t seed_;
  SelfPlayEngine engine_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<binlog::BinaryLogWriter> binary_writer_;
};

}  // namespace scribblez
