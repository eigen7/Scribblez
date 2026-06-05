// play_game: runs Scrabble games between two agents (Greedy, HastyBot, or
// Human) and writes one GCG-format game log per game to the output. A Human
// player is driven through a local web UI.
//
// Usage:
//   play_game [--player "--type=T [opts]"]... [--kwg <path>] [--seed N]
//             [--out games.gcg] [--macondo PATH] [--games N] [--verbose]
//
// Each --player spec selects a seat; repeat once per seat (defaults to two
// greedy players). The human agent's own --port / --vite-port / --web-dir
// options live inside its --player spec, e.g.
//   --player "--type=human --port=8081 --web-dir=web"
//
// For human play the engine launches the front-end's Vite dev server itself
// (npm run dev) and opens the browser at it -- you never run npm by hand. Run
// ./build.py once first to install the web dependencies.

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/gcg_writer.h"
#include "scribblez/macondo.h"
#include "scribblez/player_factory.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// Default lexicon location, relative to the current working directory. The .kwg
// binary is not committed (it encodes a copyrighted wordlist); place or symlink
// it here, or point --kwg elsewhere.
constexpr const char* kDefaultKwg = "data/lexica/NWL23.kwg";

// Persistent (across games) tally of results, indexed by *player identity*
// rather than seat -- since seats swap between games, "wins[0]" is the wins
// of the agent specified by the first --player option regardless of which
// side of the board they happened to play on for a given game.
class BatchResults {
 public:
  BatchResults(std::string name_a, std::string name_b, std::ostream& gcg_out)
      : names_{std::move(name_a), std::move(name_b)}, gcg_out_(gcg_out) {}

  // Append the game's GCG to the output, and tally win/loss/draw and turn
  // count. `player_at_seat[s]` is the persistent player index (0 or 1) that
  // sat at seat `s` in this game; used to translate the per-seat final
  // scores in `log` back into per-player tallies.
  void record(const scribblez::GameLog& log, const std::array<int, 2>& player_at_seat) {
    gcg_out_ << scribblez::game_log_to_gcg(log);
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
  long total_turns() const { return total_turns_; }

  // Per-game summary, used in verbose single-game mode.
  void print_game_summary(std::ostream& os, const scribblez::GameLog& log) const {
    os << "Final scores: " << log.final_scores[0] << " - " << log.final_scores[1] << "  ("
       << log.end_reason << ")\n"
       << "Turns: " << log.turns.size() << "\n";
  }

  // Multi-game W/L/D + throughput summary, used in verbose batch mode.
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

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;

  std::string kwg_path, out_path;
  std::vector<std::string> player_specs;
  int games = 1;
  uint64_t seed = 0;
  bool verbose = false;

  // Macondo's CLI knobs (--macondo) live on Macondo::Params; we just hand it
  // our options_description and let it register what it needs.
  scribblez::Macondo::Params macondo_params;

  po::options_description desc("play_game options");
  auto opt = desc.add_options();
  opt("help,h", "show this help message and exit");
  opt("player", po::value<std::vector<std::string>>(&player_specs)->composing(),
      "add a seat, e.g. --player \"--type=human\" --player \"--type=greedy\" "
      "(repeat once per seat; default: two greedy)");
  opt("kwg", po::value<std::string>(&kwg_path)->default_value(kDefaultKwg),
      "lexicon .kwg file to load");
  opt("seed", po::value<uint64_t>(&seed), "PRNG seed (default: hardware random)");
  opt("out", po::value<std::string>(&out_path), "write the GCG game logs here (default: stdout)");
  opt("games", po::value<int>(&games)->default_value(1),
      "play at least this many games in one process (seeds seed, seed+1, ...); "
      "humans may extend the loop via the Play Again button");
  opt("verbose,v", po::bool_switch(&verbose), "print final score and turn count to stderr");
  macondo_params.add_options(desc);

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
    return 2;
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::cout << "Player types (use --player \"--type=X [options]\"):\n\n"
              << scribblez::all_player_types_help();
    return 0;
  }
  const bool seed_given = vm.count("seed") > 0;

  // Resolve the two player seats, defaulting to two greedy players. Each spec
  // is parsed by the player factory (e.g. --player "--type=human").
  if (player_specs.empty()) {
    player_specs = {"--type=greedy", "--type=greedy"};
  }
  if (player_specs.size() != 2) {
    std::cerr << "Expected exactly two --player specs (got " << player_specs.size() << ").\n";
    return 2;
  }
  std::array<scribblez::PlayerSpec, 2> specs;
  for (int s = 0; s < 2; ++s) {
    try {
      specs[s] = scribblez::parse_player_spec(player_specs[s]);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 2;
    }
  }
  if (games < 1) {
    std::cerr << "--games must be >= 1\n";
    return 2;
  }

  // Configure Macondo. This is cheap (no subprocess yet) and unconditional;
  // the subprocess is spawned lazily the first time any agent calls
  // Macondo::instance().evaluate(...).
  scribblez::Macondo::set_params(macondo_params);

  // Resolve the seed (the lexicon path already defaulted via program_options).
  if (!seed_given) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }
  // Seed the process-wide SeedProducer so every RNG-using object constructed
  // after this point (e.g. GreedyAgent's tie-breaker) is deterministic when
  // --seed was given. Per-agent --seed options still override locally.
  scribblez::SeedProducer::instance().seed(seed);

  scribblez::Dictionary dict;
  try {
    dict = scribblez::Dictionary::load_kwg(kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << kwg_path
              << "'. Place or symlink an NWL "
                 ".kwg there, or pass --kwg <path>.\n";
    return 1;
  }
  if (verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from " << kwg_path << "\n";
    std::cerr << "Seed: " << seed << "\n";
  }

  // Construct each agent exactly once. The Human agent's ctor stands up its
  // own WebSocket server and Vite dev server, so this is the point at which
  // the browser UI comes online (for human seats).
  std::array<std::unique_ptr<scribblez::Agent>, 2> agents;
  try {
    agents[0] = scribblez::make_player(specs[0], specs[1].display_name());
    agents[1] = scribblez::make_player(specs[1], specs[0].display_name());
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  std::ofstream of;
  std::ostream* out = &std::cout;
  if (!out_path.empty()) {
    of.open(out_path);
    if (!of) {
      std::cerr << "Failed to open output file: " << out_path << "\n";
      return 1;
    }
    out = &of;
  }

  BatchResults results(specs[0].display_name(), specs[1].display_name(), *out);

  // Unified game loop. `player_at_seat[s]` is the persistent player index
  // (0 or 1, into `agents`/`specs`) currently sitting at seat s. The loop
  // plays at least `--games` games; either agent's end_game() can shorten
  // (QUIT) or extend (PLAY_AGAIN) that. Seats swap every game so the two
  // players alternate who starts; who starts game 1 is decided by the low
  // bit of the seed.
  std::array<int, 2> player_at_seat = {static_cast<int>(seed & 1ULL),
                                       static_cast<int>(1 - (seed & 1ULL))};
  uint64_t game_idx = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    scribblez::Agent& seat0 = *agents[player_at_seat[0]];
    scribblez::Agent& seat1 = *agents[player_at_seat[1]];
    scribblez::Game game(seat0, seat1, dict, seed + game_idx);
    game.play();
    const scribblez::GameLog& log = game.log();
    results.record(log, player_at_seat);
    if (verbose) results.print_game_summary(std::cerr, log);

    auto r0 = seat0.end_game(game, 0);
    auto r1 = seat1.end_game(game, 1);
    bool quit = r0.action == scribblez::EndGameAction::QUIT ||
                r1.action == scribblez::EndGameAction::QUIT;
    bool play_again = r0.action == scribblez::EndGameAction::PLAY_AGAIN ||
                      r1.action == scribblez::EndGameAction::PLAY_AGAIN;
    if (quit) break;
    if (!play_again && results.games_played() >= games) break;

    std::swap(player_at_seat[0], player_at_seat[1]);
    ++game_idx;
  }

  if (verbose && results.games_played() > 1) {
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    results.print_batch_summary(std::cerr, secs);
  }
  return 0;
}
