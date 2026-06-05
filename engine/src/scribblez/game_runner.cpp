#include "scribblez/game_runner.h"

#include "scribblez/exception.h"
#include "scribblez/gcg_writer.h"

#include <boost/program_options.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <utility>

namespace scribblez {

// --------------------------- Results -------------------------------------

class GameRunner::Results {
 public:
  Results(std::array<std::string, 2> names, std::ostream& gcg_out)
      : names_(std::move(names)), gcg_out_(gcg_out) {}

  // Append the game's GCG to the output, and tally win/loss/draw and turn
  // count. `player_at_seat[s]` is the persistent player index (0 or 1) that
  // sat at seat `s` in this game; used to translate per-seat final scores
  // back into per-player tallies.
  void record(const GameLog& log, const std::array<int, 2>& player_at_seat) {
    gcg_out_ << game_log_to_gcg(log);
    total_turns_ += static_cast<long>(log.turns.size());
    ++games_played_;
    if (log.final_scores[0] == log.final_scores[1]) {
      ++draws_;
    } else {
      int winning_seat = log.final_scores[0] > log.final_scores[1] ? 0 : 1;
      ++wins_[player_at_seat[winning_seat]];
    }
  }

  int games_played() const { return games_played_; }

  void print_game_summary(std::ostream& os, const GameLog& log) const {
    os << "Final scores: " << log.final_scores[0] << " - " << log.final_scores[1] << "  ("
       << log.end_reason << ")\n"
       << "Turns: " << log.turns.size() << "\n";
  }

  void print_batch_summary(std::ostream& os, double elapsed_secs) const {
    os << games_played_ << " games in " << elapsed_secs << "s -> "
       << (games_played_ / elapsed_secs) << " games/s, " << total_turns_ << " turns -> "
       << (total_turns_ / elapsed_secs) << " moves/s\n"
       << names_[0] << " W/L/D vs " << names_[1] << ": " << wins_[0] << " / " << wins_[1] << " / "
       << draws_ << "\n";
  }

 private:
  std::array<std::string, 2> names_;
  std::ostream& gcg_out_;
  std::array<int, 2> wins_ = {0, 0};
  int draws_ = 0;
  int games_played_ = 0;
  long total_turns_ = 0;
};

// --------------------------- Params --------------------------------------

void GameRunner::Params::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()                                                                  //
      ("kwg", po::value<std::string>(&kwg_path)->default_value(kwg_path),             //
       "lexicon .kwg file to load")                                                   //
      ("games", po::value<int>(&games)->default_value(games),                         //
       "play at least this many games in one process (seeds seed, seed+1, ...); "
       "humans may extend the loop via the Play Again button")                        //
      ("out", po::value<std::string>(&out_path),                                      //
       "write the GCG game logs here (default: stdout)")                              //
      ("verbose,v", po::bool_switch(&verbose),                                        //
       "print final score and turn count to stderr");
}

// --------------------------- ctor / run ----------------------------------

GameRunner::GameRunner(const Params& params, PlayerFactory::Players players, uint64_t seed)
    : params_(params),
      agents_(std::move(players.agents)),
      display_names_(std::move(players.display_names)),
      seed_(seed),
      out_(&std::cout) {
  if (params_.games < 1) {
    std::cerr << "Error: --games must be >= 1\n";
    throw Exception("--games must be >= 1");
  }
  try {
    dict_ = Dictionary::load_kwg(params_.kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << params_.kwg_path
              << "'. Place or symlink an NWL .kwg there, or pass --kwg <path>.\n";
    throw Exception(e.what());
  }
  if (!params_.out_path.empty()) {
    of_.open(params_.out_path);
    if (!of_) {
      std::cerr << "Error: failed to open output file: " << params_.out_path << "\n";
      throw Exception("failed to open output file: " + params_.out_path);
    }
    out_ = &of_;
  }
  if (params_.verbose) {
    std::cerr << "Loaded KWG (" << dict_.num_nodes() << " nodes) from " << params_.kwg_path << "\n"
              << "Seed: " << seed_ << "\n";
  }
  results_ = std::make_unique<Results>(display_names_, *out_);
}

GameRunner::~GameRunner() = default;

bool GameRunner::play_one_game(std::array<int, 2>& player_at_seat, uint64_t game_idx) {
  Agent& seat0 = *agents_[player_at_seat[0]];
  Agent& seat1 = *agents_[player_at_seat[1]];
  Game game(seat0, seat1, dict_, seed_ + game_idx);
  game.play();
  const GameLog& log = game.log();
  results_->record(log, player_at_seat);
  if (params_.verbose) results_->print_game_summary(std::cerr, log);

  auto r0 = seat0.end_game(game, 0);
  auto r1 = seat1.end_game(game, 1);
  bool quit = r0.action == EndGameAction::QUIT || r1.action == EndGameAction::QUIT;
  bool play_again =
      r0.action == EndGameAction::PLAY_AGAIN || r1.action == EndGameAction::PLAY_AGAIN;
  if (quit) return false;
  if (!play_again && results_->games_played() >= params_.games) return false;
  return true;
}

void GameRunner::run() {
  // Seats swap every game so the two players alternate who starts; who
  // starts game 1 is decided by the low bit of the seed.
  std::array<int, 2> player_at_seat = {static_cast<int>(seed_ & 1ULL),
                                       static_cast<int>(1 - (seed_ & 1ULL))};
  uint64_t game_idx = 0;
  auto t0 = std::chrono::steady_clock::now();
  while (play_one_game(player_at_seat, game_idx)) {
    std::swap(player_at_seat[0], player_at_seat[1]);
    ++game_idx;
  }
  if (params_.verbose && results_->games_played() > 1) {
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    results_->print_batch_summary(std::cerr, secs);
  }
}

}  // namespace scribblez
