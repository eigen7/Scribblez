// play_game: runs a single Greedy vs Greedy Scrabble game and writes a JSON log.
//
// Usage:
//   play_game --dict <path> [--seed N] [--out game.json] [--verbose]

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/json_writer.h"

namespace {

// Default lexicon location, relative to the current working directory. The .kwg
// binary is not committed (it encodes a copyrighted wordlist); place or symlink
// it here, or point --kwg elsewhere.
constexpr const char* kDefaultKwg = "data/lexica/NWL23.kwg";

void usage() {
  std::cerr << "Usage: play_game [--kwg <lexicon.kwg>] [--seed N] "
               "[--out game.json] [--verbose]\n"
               "  --kwg defaults to "
            << kDefaultKwg << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string kwg_path;
  std::string out_path;
  uint64_t seed = 0;
  bool seed_given = false;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing argument for " << flag << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--kwg" || a == "--dict") kwg_path = need(a);
    else if (a == "--out") out_path = need(a);
    else if (a == "--seed") { seed = std::stoull(need(a)); seed_given = true; }
    else if (a == "--verbose" || a == "-v") verbose = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else {
      std::cerr << "Unknown argument: " << a << "\n";
      usage();
      return 2;
    }
  }

  // Resolve the lexicon: explicit flag, else the default path.
  if (kwg_path.empty()) kwg_path = kDefaultKwg;
  if (!seed_given) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }

  scribblez::Dictionary dict;
  try {
    dict = scribblez::Dictionary::load_kwg(kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << kwg_path << "'. Place or symlink an NWL "
                 ".kwg there, or pass --kwg <path>.\n";
    return 1;
  }
  if (verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from " << kwg_path << "\n";
    std::cerr << "Seed: " << seed << "\n";
  }

  scribblez::Game game(std::make_unique<scribblez::GreedyAgent>(),
                       std::make_unique<scribblez::GreedyAgent>(),
                       dict, seed);
  game.play();

  const auto& log = game.log();
  std::string json = scribblez::game_log_to_json(log);
  if (out_path.empty()) {
    std::cout << json;
  } else {
    std::ofstream of(out_path);
    if (!of) {
      std::cerr << "Failed to open output file: " << out_path << "\n";
      return 1;
    }
    of << json;
  }

  if (verbose) {
    std::cerr << "Final scores: " << log.final_scores[0] << " - "
              << log.final_scores[1] << "  (" << log.end_reason << ")\n";
    std::cerr << "Turns: " << log.turns.size() << "\n";
  }
  return 0;
}
