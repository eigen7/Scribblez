// Offline harvester for the large position-evaluation Monte-Carlo test set.
//
// Plays HastyBot-vs-HastyBot games and, from each game, samples one
// "penultimate-bingo" position: a turn where a player bingoed (placed all 7 rack
// tiles) with a full bag to refill from, immediately followed by the opponent's
// tile placement. The game is truncated one move after the bingo and written as a
// GCG from the final mover's POV -- exactly the input `monte_carlo_sim_tool`
// scores. Because the bingoer's post-bingo rack is a clean full draw, the sim can
// sample the opponent's rack uniformly from the unseen pool with no leave model.
//
// The emitted GCG keeps each move line's rack_before (the sim reads the final
// mover's leave from it) but writes NO #Rack pragmas and no end-of-game rack
// adjustments, so the bingoer's actual drawn rack never appears.
//
// Output: positions/<lexicon>/<dataset-name>/part-NNN.gcgs, each a concatenation
// of up to --per-file GCG blocks. Every block begins with `#character-encoding`,
// the record boundary the Python wrapper splits on before scoring.

#include "agent/macondo_bot.h"
#include "data/gcg_writer.h"
#include "game/game.h"
#include "game/move.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "selfplay/game_runner.h"
#include "util/cli.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {
namespace {

namespace fs = std::filesystem;

// A qualifying bingo at turn `i`: a 7-tile placement (rack emptied) whose refill
// draws from a full bag, immediately followed by a tile placement -- so the
// recorded final move (turn i+1) is a PLAY, as parse_monte_carlo_position needs.
bool is_qualifying_bingo(const GameLogStorage& log, int i) {
  if (i + 1 >= static_cast<int>(log.turns.size())) return false;
  const TurnRecord& bingo = log.turns[i];
  const bool bingo_ok = bingo.move.type() == MoveType::PLAY &&
                        bingo.move.num_glyphs() == RACK_SIZE && bingo.bag_size_before >= RACK_SIZE;
  return bingo_ok && log.turns[i + 1].move.type() == MoveType::PLAY;
}

// The qualifying-bingo turn indices in a played-out game.
std::vector<int> qualifying_bingos(const GameLogStorage& log) {
  std::vector<int> out;
  for (int i = 0; i < static_cast<int>(log.turns.size()); ++i)
    if (is_qualifying_bingo(log, i)) out.push_back(i);
  return out;
}

// Truncate the log to end one move after the bingo, and set final_scores to the
// last kept cumulative so the writer emits no END_RACK adjustment lines.
GameLogStorage truncate_to_post_bingo(GameLogStorage log, int bingo_idx) {
  log.turns.resize(bingo_idx + 2);  // keep through turn bingo_idx + 1
  log.final_scores = log.turns.back().cumulative_scores;
  log.final_racks = {};
  log.end_reason.clear();
  return log;
}

// The GCG for one harvested position: rack_before kept per line, no #Rack pragmas.
std::string harvested_gcg(GameLogStorage log, int bingo_idx, uint64_t seed,
                          const std::string& lexicon) {
  const GameLogStorage truncated = truncate_to_post_bingo(std::move(log), bingo_idx);
  GcgWriteOptions options;
  options.lexicon_name = lexicon;
  options.notes = {"Harvested penultimate-bingo position (game seed " + std::to_string(seed) + ")"};
  return game_log_to_gcg(truncated.view(), options);
}

// Play game `seed` and, if it has any qualifying bingo, return the GCG for one
// sampled uniformly (RNG seeded by the game, for reproducibility). Empty if none.
std::string harvest_from_game(HastyBotAgent& a0, HastyBotAgent& a1, const Dictionary& dict,
                              uint64_t seed, const std::string& lexicon) {
  Game game(a0, a1, dict, seed);
  game.play();
  GameLogStorage log = game.extract_log();
  const std::vector<int> bingos = qualifying_bingos(log);
  if (bingos.empty()) return {};
  std::mt19937_64 rng(seed);
  const int pick = bingos[std::uniform_int_distribution<std::size_t>(0, bingos.size() - 1)(rng)];
  return harvested_gcg(std::move(log), pick, seed, lexicon);
}

std::string part_filename(int part) {
  std::ostringstream name;
  name << "part-" << std::setw(3) << std::setfill('0') << part << ".gcgs";
  return name.str();
}

// Write the collected GCG blocks into per_file-sized bundle files.
void write_bundles(const fs::path& dir, const std::vector<std::string>& gcgs, int per_file) {
  const int num_parts = (static_cast<int>(gcgs.size()) + per_file - 1) / per_file;
  for (int part = 0; part < num_parts; ++part) {
    std::ofstream os(dir / part_filename(part));
    const int end = std::min(static_cast<int>(gcgs.size()), (part + 1) * per_file);
    for (int k = part * per_file; k < end; ++k) os << gcgs[k];
  }
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string dataset_name = "position-eval-test-dataset-large";
    int count = 1000;
    int per_file = 100;
    long base_seed = 1000000;

    po::options_description desc("harvest_bingo_positions_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "dataset-name", po::value<std::string>(&dataset_name)->default_value(dataset_name),
      "dataset under positions/<lexicon>/; bundle files are written here")(
      "count", po::value<int>(&count)->default_value(count), "number of positions to harvest")(
      "per-file", po::value<int>(&per_file)->default_value(per_file), "GCG blocks per bundle file")(
      "seed", po::value<long>(&base_seed)->default_value(base_seed),
      "base game seed (scans base_seed, base_seed+1, ...; a reserved range disjoint from "
      "training)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    const std::string& lexicon = scribblez::Lexicon::instance().name();
    const std::filesystem::path dir = std::filesystem::path("positions") / lexicon / dataset_name;
    std::filesystem::create_directories(dir);

    scribblez::HastyBotAgent::Params p0;
    p0.name = "Hasty_1";
    scribblez::HastyBotAgent::Params p1;
    p1.name = "Hasty_2";
    scribblez::HastyBotAgent a0(p0), a1(p1);

    std::vector<std::string> gcgs;
    long seed = base_seed;
    long scanned = 0;
    while (static_cast<int>(gcgs.size()) < count) {
      std::string gcg =
        scribblez::harvest_from_game(a0, a1, dict, static_cast<uint64_t>(seed), lexicon);
      if (!gcg.empty()) gcgs.push_back(std::move(gcg));
      ++seed;
      ++scanned;
    }
    scribblez::write_bundles(dir, gcgs, per_file);

    std::cerr << "Harvested " << gcgs.size() << " positions from " << scanned << " games into "
              << dir.string() << " (" << ((count + per_file - 1) / per_file) << " bundle files)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
