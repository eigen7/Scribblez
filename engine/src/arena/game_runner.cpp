#include "arena/game_runner.h"

#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "game/game_log.h"
#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/seed_producer.h"
#include "util/string.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace scribblez {

// Games per .slog file. One file is also the unit a self-play generator stages
// as a chunk, so this fixes the chunk size fleet-wide.
constexpr int kGamesPerFile = 1000;

// --------------------------- Results -------------------------------------

class GameRunner::Results {
 public:
  // `out`, if open, receives one JSON line per finished game.
  Results(std::array<std::string, 2> names, std::ofstream out)
      : names_(std::move(names)), out_(std::move(out)) {}

  // `seats[s]` is the persistent player index that sat at seat s. Thread-safe.
  void record(const GameLog& log, const std::array<int, 2>& seats, bool verbose) {
    int winning_seat = -1;
    if (log.final_scores[0] != log.final_scores[1])
      winning_seat = log.final_scores[0] > log.final_scores[1] ? 0 : 1;

    std::lock_guard<std::mutex> lock(mutex_);
    total_turns_ += log.num_records;
    ++games_played_;
    if (winning_seat < 0) {
      ++draws_;
    } else {
      ++wins_[seats[winning_seat]];
    }
    if (out_.is_open()) write_result_line(log, seats);
    if (verbose) {
      std::cerr << "Final scores: " << log.final_scores[0] << " - " << log.final_scores[1] << "  ("
                << log.end_reason << ")\n"
                << "Turns: " << log.num_records << "\n";
    }
  }

  // Thread-safe.
  int games_played() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return games_played_;
  }

  // A live progress line: games done out of `total`, throughput, and ETA.
  // Thread-safe.
  void print_progress(std::ostream& os, double elapsed_secs, uint64_t total) const {
    int done;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done = games_played_;
    }
    const double rate = elapsed_secs > 0 ? done / elapsed_secs : 0.0;
    const double pct = total > 0 ? 100.0 * done / double(total) : 0.0;
    const double eta = rate > 0 ? (double(total) - done) / rate : 0.0;
    os << "[progress] " << done << "/" << total << " games (" << std::fixed << std::setprecision(1)
       << pct << "%) | " << std::setprecision(2) << rate << " games/s | elapsed "
       << util::fmt_dur(elapsed_secs) << " | ETA " << util::fmt_dur(eta) << "\n";
  }

  void print_batch_summary(std::ostream& os, double elapsed_secs) const {
    os << games_played_ << " games in " << elapsed_secs << "s -> " << (games_played_ / elapsed_secs)
       << " games/s, " << total_turns_ << " turns -> " << (total_turns_ / elapsed_secs)
       << " moves/s\n"
       << names_[0] << " W/L/D vs " << names_[1] << ": " << wins_[0] << " / " << wins_[1] << " / "
       << draws_ << "\n";
  }

 private:
  // One self-contained line per game, flushed so a supervising process can
  // read results as they stream. Caller holds `mutex_`.
  void write_result_line(const GameLog& log, const std::array<int, 2>& seats) {
    boost::json::object o;
    o["seed"] = log.seed;
    o["seat_players"] = boost::json::array{seats[0], seats[1]};
    o["seat_scores"] = boost::json::array{log.final_scores[0], log.final_scores[1]};
    o["turns"] = log.num_records;
    o["end_reason"] = log.end_reason;
    out_ << boost::json::serialize(o) << "\n" << std::flush;
  }

  mutable std::mutex mutex_;
  std::array<std::string, 2> names_;
  std::ofstream out_;
  std::array<int, 2> wins_ = {0, 0};
  int draws_ = 0;
  int games_played_ = 0;
  long total_turns_ = 0;
};

// --------------------------- Params --------------------------------------

void GameRunner::Params::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()                                         //
    ("games", po::value<int>(&games)->default_value(games),  //
     "play at least this many games in one process (seeds seed, seed+1, ...); "
     "humans may extend the loop via the Play Again button")                         //
    ("log-dir", po::value<std::string>(&log_dir),                                    //
     "directory to write one <timestamp>.gcg log per game (omit to suppress logs)")  //
    ("binary-log-dir", po::value<std::string>(&binary_log_dir),                      //
     "directory to write batched binary .slog files (one file per "
     "kGamesPerFile games; every eligible position is written -- "
     "train-time sampling is done by the DataLoader)")               //
    ("threads,t", po::value<int>(&threads)->default_value(threads),  //
     "number of parallel game threads (default: all logical processors; "
     "downgraded to 1 when a player does not support parallelism, e.g. a "
     "human seat)")  //
    ("random-handicap-max",
     po::value<int>(&random_handicap_max)->default_value(random_handicap_max),
     "if > 0, each game gifts a randomly chosen player a head-start of P "
     "points, with P drawn uniformly from [0, this value]")  //
    ("random-opening-mean",
     po::value<double>(&random_opening_mean)->default_value(random_opening_mean),
     "if > 0, each game opens with K uniformly-random plies before the agents "
     "take over, with K drawn per game from an exponential of this mean "
     "(rounded to the nearest integer); positions before the last random ply "
     "are excluded from the .slog training-eligible region")  //
    ("respect-projections",
     po::value<bool>(&respect_projections)->default_value(respect_projections),
     "play out an agent's proven remaining-game projection directly instead of "
     "prompting agents turn by turn (the endgame solver's break-out)")  //
    ("face-up-leaves", po::bool_switch(&face_up_leaves),
     "play face-up-leaves Scrabble: the tiles each player retains from their "
     "move are public until they move again, with only their replenishment "
     "draws hidden (docs/roadmap.md)")  //
    ("paired", po::bool_switch(&paired),
     "play games in mirrored pairs: games 2k and 2k+1 share one game seed "
     "with the seats swapped, so per-seed tile luck cancels out of paired "
     "comparisons (requires an even --games)")  //
    ("results-file", po::value<std::string>(&results_file),
     "write one JSON line per finished game ({seed, seat_players, "
     "seat_scores, turns, end_reason}), flushed as games complete")  //
    ("progress-secs", po::value<int>(&progress_secs)->default_value(progress_secs),
     "print a games-done/rate/ETA progress line to stderr every this many "
     "seconds during the parallel batch loop (0 disables)")  //
    ("verbose,v", po::bool_switch(&verbose),                 //
     "print final score and turn count to stderr");
}

// --------------------------- monitor -------------------------------------

void GameRunner::run_progress_monitor(const std::atomic<bool>& done,
                                      std::chrono::steady_clock::time_point t0,
                                      uint64_t total) const {
  const int ticks = params_.progress_secs * 10;  // poll the stop flag at 10 Hz
  while (!done.load(std::memory_order_acquire)) {
    for (int i = 0; i < ticks && !done.load(std::memory_order_acquire); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (done.load(std::memory_order_acquire)) break;
    const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    results_->print_progress(std::cerr, secs, total);
  }
}

// --------------------------- ctor / run ----------------------------------

GameRunner::GameRunner(const Params& params, const PlayerFactory::Params& player_params)
    : params_(params),
      seed_(SeedProducer::instance().next()),
      engine_(GameEngine::Params{params.threads, seed_, params.random_handicap_max,
                                 params.random_opening_mean, params.respect_projections,
                                 params.face_up_leaves},
              player_params) {
  if (params_.games < 1) throw util::CleanException("--games must be >= 1");
  if (params_.paired && params_.games % 2 != 0) {
    throw util::CleanException("--paired requires an even --games");
  }
  // Force the lexicon load now so any I/O error surfaces at construction
  // time (rather than mid-game), and so the verbose summary below has the
  // node count to report.
  const Dictionary& dict = load_dictionary_or_throw();
  if (!params_.log_dir.empty()) {
    std::error_code ec;
    if (!std::filesystem::is_directory(params_.log_dir, ec)) {
      throw util::CleanException("--log-dir '{}' is not a directory", params_.log_dir);
    }
  }
  if (!params_.binary_log_dir.empty()) {
    std::error_code ec;
    if (!std::filesystem::is_directory(params_.binary_log_dir, ec)) {
      throw util::CleanException("--binary-log-dir '{}' is not a directory",
                                 params_.binary_log_dir);
    }
    binary_writer_ = std::make_unique<binlog::BinaryLogWriter>(
      params_.binary_log_dir, kGamesPerFile,
      params_.face_up_leaves ? binlog::kFlagFaceUpLeaves : 0);
  }
  std::ofstream results_out;
  if (!params_.results_file.empty()) {
    results_out.open(params_.results_file, std::ios::trunc);
    if (!results_out) {
      throw util::CleanException("cannot open --results-file '{}'", params_.results_file);
    }
  }
  if (params_.verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from "
              << Lexicon::instance().kwg_path() << "\n"
              << "Seed: " << seed_ << "\n";
  }
  results_ = std::make_unique<Results>(engine_.player_names(), std::move(results_out));
}

GameRunner::~GameRunner() = default;

void GameRunner::on_game(GameLogStorage&& storage, const std::array<int, 2>& seats) {
  // Read the finished game through a view for the gcg/tally consumers, then hand
  // ownership to the binary writer (a move). Take the view BEFORE the move; do
  // not touch it afterward (the move invalidates its pointers).
  const GameLog log = storage.view();

  if (!params_.log_dir.empty()) {
    std::string path = params_.log_dir + "/" + std::to_string(util::get_unique_id()) + ".gcg";
    std::ofstream f(path);
    if (f) {
      write_game_log_gcg(log, f);
    } else {
      std::cerr << "Warning: failed to open log file: " << path << "\n";
    }
  }

  results_->record(log, seats, params_.verbose);

  if (binary_writer_) {
    binary_writer_->append(std::move(storage));
  }
}

void GameRunner::run() {
  auto t0 = std::chrono::steady_clock::now();

  if (engine_.num_threads() == 1) {
    // Serial mode: supports PLAY_AGAIN / QUIT signalling from agents.
    // Seats swap every game so the two players alternate who starts; who
    // starts game 1 is decided by the low bit of the seed.
    std::array<int, 2> player_at_seat = {int(seed_ & 1ULL), int(1 - (seed_ & 1ULL))};
    uint64_t game_idx = 0;
    while (true) {
      auto [a0, a1] = engine_.play(0, player_at_seat, seed_index(game_idx), *this);
      bool quit = a0 == EndGameAction::QUIT || a1 == EndGameAction::QUIT;
      bool play_again = a0 == EndGameAction::PLAY_AGAIN || a1 == EndGameAction::PLAY_AGAIN;
      if (quit) break;
      if (!play_again && results_->games_played() >= params_.games) break;
      std::swap(player_at_seat[0], player_at_seat[1]);
      ++game_idx;
    }
  } else {
    // Parallel mode: play exactly params_.games games across a thread pool.
    // Each game's seat assignment alternates with game_idx so overall balance
    // is preserved across any interleaving of threads.
    std::atomic<uint64_t> next_game{0};
    const uint64_t total = params_.games;

    // Optional monitor thread: prints a games-done/rate/ETA line every
    // progress_secs seconds. Useful for long self-play batches (e.g. the
    // all-moves neural agent), where no .slog file flushes until a full
    // .slog chunk completes, so this is the only sign of life.
    std::atomic<bool> done{false};
    std::thread monitor;
    if (params_.progress_secs > 0) {
      monitor = std::thread(&GameRunner::run_progress_monitor, this, std::ref(done), t0, total);
    }

    std::vector<std::thread> workers;
    workers.reserve(size_t(engine_.num_threads()));
    for (int t = 0; t < engine_.num_threads(); ++t) {
      workers.emplace_back([&, t]() {
        while (true) {
          uint64_t idx = next_game.fetch_add(1, std::memory_order_acq_rel);
          if (idx >= total) break;
          int seat0_player = (seed_ + idx) & 1ULL;
          std::array<int, 2> seats = {seat0_player, 1 - seat0_player};
          engine_.play(t, seats, seed_index(idx), *this);
        }
      });
    }
    for (auto& w : workers) w.join();

    done.store(true, std::memory_order_release);
    if (monitor.joinable()) monitor.join();
  }

  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  results_->print_batch_summary(std::cerr, secs);
}

}  // namespace scribblez
