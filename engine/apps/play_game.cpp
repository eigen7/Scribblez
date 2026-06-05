// play_game: runs Scrabble games between two agents (Greedy, HastyBot, or
// Human) and writes one GCG-format game log per game to the output. A Human
// player is driven through a local web UI.
//
// Usage:
//   play_game [--player "--type=T [opts]"]... [--kwg PATH] [--seed N]
//             [--out PATH] [--macondo PATH] [--games N] [--verbose]
//
// Each --player spec selects a seat; repeat once per seat (defaults to two
// greedy players). The human agent's own --port / --vite-port / --web-dir
// options live inside its --player spec, e.g.
//   --player "--type=human --port=8081 --web-dir=web"
//
// For human play the engine launches the front-end's Vite dev server itself
// (npm run dev) and opens the browser at it -- you never run npm by hand. Run
// ./build.py once first to install the web dependencies.

#include "scribblez/dictionary.h"
#include "scribblez/game_runner.h"
#include "scribblez/macondo.h"
#include "scribblez/player_factory.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

namespace {

// Default lexicon location, relative to the current working directory. The .kwg
// binary is not committed (it encodes a copyrighted wordlist); place or symlink
// it here, or point --kwg elsewhere.
constexpr const char* kDefaultKwg = "data/lexica/NWL23.kwg";

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;

  // Each subsystem owns its own Params + add_options() so this top-level
  // function never has to know which knobs belong to whom.
  scribblez::Macondo::Params macondo_params;
  scribblez::PlayerFactory::Params player_params;
  scribblez::GameRunner::Params runner_params;

  std::string kwg_path;
  uint64_t seed_value = 0;
  bool seed_given_flag = false;

  po::options_description desc("play_game options");
  desc.add_options()                                                                       //
      ("help,h", "show this help message and exit")                                        //
      ("kwg", po::value<std::string>(&kwg_path)->default_value(kDefaultKwg),               //
       "lexicon .kwg file to load")                                                        //
      ("seed",                                                                             //
       po::value<uint64_t>(&seed_value)->notifier([&](uint64_t) { seed_given_flag = true; }),
       "PRNG seed (default: hardware random)");
  macondo_params.add_options(desc);
  player_params.add_options(desc);
  runner_params.add_options(desc);

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
              << scribblez::PlayerFactory::all_player_types_help();
    return 0;
  }

  scribblez::Macondo::set_params(macondo_params);

  // Seed the process-wide SeedProducer (random-device fallback handled
  // inside seed()). All RNG-using objects constructed after this point are
  // deterministic for a given --seed.
  std::optional<uint64_t> requested = seed_given_flag ? std::optional{seed_value} : std::nullopt;
  uint64_t seed = scribblez::SeedProducer::instance().seed(requested);

  scribblez::Dictionary dict;
  try {
    dict = scribblez::Dictionary::load_kwg(kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << kwg_path
              << "'. Place or symlink an NWL .kwg there, or pass --kwg <path>.\n";
    return 1;
  }
  if (runner_params.verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from " << kwg_path << "\n"
              << "Seed: " << seed << "\n";
  }

  try {
    auto players = scribblez::PlayerFactory::make_players(player_params);
    scribblez::GameRunner runner(runner_params, std::move(players), dict, seed);
    runner.run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
