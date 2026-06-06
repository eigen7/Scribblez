#include "scribblez/game_runner.h"

#include "scribblez/binary_log.h"
#include "scribblez/dictionary.h"
#include "scribblez/exception.h"
#include "scribblez/gcg_writer.h"
#include "scribblez/lexicon.h"
#include "scribblez/player_factory.h"
#include "scribblez/seed_producer.h"
#include "scribblez/unique_id.h"

#include <boost/program_options.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace scribblez {

// --------------------------- Results -------------------------------------

class GameRunner::Results {
 public:
  explicit Results(std::array<std::string, 2> names) : names_(std::move(names)) {}

  // Append the game tally. `seats[s]` is the persistent player index (0 or 1)
  // that sat at seat `s` in this game. Thread-safe.
  void record(const GameLog& log, const std::array<int, 2>& seats, bool verbose) {
    int winning_seat = -1;
    if (log.final_scores[0] != log.final_scores[1])
      winning_seat = log.final_scores[0] > log.final_scores[1] ? 0 : 1;

    std::lock_guard<std::mutex> lock(mutex_);
    total_turns_ += static_cast<long>(log.turns.size());
    ++games_played_;
    if (winning_seat < 0) {
      ++draws_;
    } else {
      ++wins_[seats[winning_seat]];
    }
    if (verbose) {
      std::cerr << "Final scores: " << log.final_scores[0] << " - " << log.final_scores[1] << "  ("
                << log.end_reason << ")\n"
                << "Turns: " << log.turns.size() << "\n";
    }
  }

  // Returns the number of games recorded so far. Thread-safe.
  int games_played() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return games_played_;
  }

  void print_batch_summary(std::ostream& os, double elapsed_secs) const {
    os << games_played_ << " games in " << elapsed_secs << "s -> " << (games_played_ / elapsed_secs)
       << " games/s, " << total_turns_ << " turns -> " << (total_turns_ / elapsed_secs)
       << " moves/s\n"
       << names_[0] << " W/L/D vs " << names_[1] << ": " << wins_[0] << " / " << wins_[1] << " / "
       << draws_ << "\n";
  }

 private:
  mutable std::mutex mutex_;
  std::array<std::string, 2> names_;
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
     "--games-per-file games; positions inside are sampled for training)")                    //
    ("games-per-file", po::value<int>(&games_per_file)->default_value(games_per_file),        //
     "games per .slog file (only used with --binary-log-dir)")                                //
    ("samples-per-game", po::value<int>(&samples_per_game)->default_value(samples_per_game),  //
     "sampled positions per game written to .slog (chosen uniformly from all "
     "pre-move positions plus post-move positions of PLAY turns)")   //
    ("threads,t", po::value<int>(&threads)->default_value(threads),  //
     "number of parallel game threads (>1 requires all players to support "
     "parallelism, i.e. no human players)")   //
    ("verbose,v", po::bool_switch(&verbose),  //
     "print final score and turn count to stderr");
}

// --------------------------- ctor / run ----------------------------------

GameRunner::GameRunner(const Params& params, const PlayerFactory::Params& player_params)
    : params_(params), seed_(SeedProducer::instance().next()) {
  if (params_.games < 1) {
    std::cerr << "Error: --games must be >= 1\n";
    throw Exception("--games must be >= 1");
  }
  if (params_.threads < 1) {
    std::cerr << "Error: --threads must be >= 1\n";
    throw Exception("--threads must be >= 1");
  }
  // Build the first pair to check parallelism support before creating the rest.
  agents_.push_back(PlayerFactory::make_players(player_params, /*thread_id=*/0));
  bool parallel_ok = agents_[0][0]->supports_parallelism() && agents_[0][1]->supports_parallelism();
  if (!parallel_ok && params_.threads > 1) {
    std::cerr << "Warning: a player does not support parallelism; running single-threaded.\n";
    params_.threads = 1;
  }
  for (int i = 1; i < params_.threads; ++i) {
    agents_.push_back(PlayerFactory::make_players(player_params, /*thread_id=*/i));
  }
  // Force the lexicon load now so any I/O error surfaces at construction
  // time (rather than mid-game), and so the verbose summary below has the
  // node count to report.
  const Dictionary* dict = nullptr;
  try {
    dict = &Lexicon::instance().dict();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "Lexicon '" << Lexicon::instance().name() << "' is not installed at "
              << Lexicon::instance().kwg_path() << ".\n"
              << "Run setup_wizard.py outside the Docker container to install it.\n";
    throw Exception(e.what());
  }
  if (!params_.log_dir.empty()) {
    std::error_code ec;
    if (!std::filesystem::is_directory(params_.log_dir, ec)) {
      std::cerr << "Error: --log-dir '" << params_.log_dir << "' is not a directory\n";
      throw Exception("--log-dir is not a directory: " + params_.log_dir);
    }
  }
  if (!params_.binary_log_dir.empty()) {
    std::error_code ec;
    if (!std::filesystem::is_directory(params_.binary_log_dir, ec)) {
      std::cerr << "Error: --binary-log-dir '" << params_.binary_log_dir
                << "' is not a directory\n";
      throw Exception("--binary-log-dir is not a directory: " + params_.binary_log_dir);
    }
    if (params_.games_per_file < 1) {
      std::cerr << "Error: --games-per-file must be >= 1\n";
      throw Exception("--games-per-file must be >= 1");
    }
    if (params_.samples_per_game < 1) {
      std::cerr << "Error: --samples-per-game must be >= 1\n";
      throw Exception("--samples-per-game must be >= 1");
    }
    binary_writer_ = std::make_unique<binlog::BinaryLogWriter>(
      params_.binary_log_dir, params_.games_per_file, params_.samples_per_game);
  }
  if (params_.verbose) {
    std::cerr << "Loaded KWG (" << dict->num_nodes() << " nodes) from "
              << Lexicon::instance().kwg_path() << "\n"
              << "Seed: " << seed_ << "\n";
  }
  results_ = std::make_unique<Results>(
    std::array<std::string, 2>{agents_[0][0]->name(), agents_[0][1]->name()});
}

GameRunner::~GameRunner() = default;

std::pair<EndGameAction, EndGameAction> GameRunner::play_one_game(int thread_idx,
                                                                  const std::array<int, 2>& seats,
                                                                  uint64_t game_idx) {
  Agent& seat0 = *agents_[thread_idx][seats[0]];
  Agent& seat1 = *agents_[thread_idx][seats[1]];
  Game game(seat0, seat1, Lexicon::instance().dict(), seed_ + game_idx);
  game.play();
  const GameLog& log = game.log();

  if (!params_.log_dir.empty()) {
    std::string path = params_.log_dir + "/" + std::to_string(get_unique_id()) + ".gcg";
    std::ofstream f(path);
    if (f) {
      write_game_log_gcg(log, f);
    } else {
      std::cerr << "Warning: failed to open log file: " << path << "\n";
    }
  }

  if (binary_writer_) {
    binary_writer_->append(log);
  }

  results_->record(log, seats, params_.verbose);

  auto r0 = seat0.end_game(game, 0);
  auto r1 = seat1.end_game(game, 1);
  return {r0.action, r1.action};
}

void GameRunner::run() {
  auto t0 = std::chrono::steady_clock::now();

  if (agents_.size() == 1) {
    // Serial mode: supports PLAY_AGAIN / QUIT signalling from agents.
    // Seats swap every game so the two players alternate who starts; who
    // starts game 1 is decided by the low bit of the seed.
    std::array<int, 2> player_at_seat = {static_cast<int>(seed_ & 1ULL),
                                         static_cast<int>(1 - (seed_ & 1ULL))};
    uint64_t game_idx = 0;
    while (true) {
      auto [a0, a1] = play_one_game(0, player_at_seat, game_idx);
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
    const uint64_t total = static_cast<uint64_t>(params_.games);
    std::vector<std::thread> workers;
    workers.reserve(agents_.size());
    for (int t = 0; t < static_cast<int>(agents_.size()); ++t) {
      workers.emplace_back([&, t]() {
        while (true) {
          uint64_t idx = next_game.fetch_add(1, std::memory_order_acq_rel);
          if (idx >= total) break;
          int seat0_player = static_cast<int>((seed_ + idx) & 1ULL);
          std::array<int, 2> seats = {seat0_player, 1 - seat0_player};
          play_one_game(t, seats, idx);
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  if (params_.verbose && results_->games_played() > 1) {
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    results_->print_batch_summary(std::cerr, secs);
  }
}

}  // namespace scribblez
