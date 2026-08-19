// Offline harvester for the large position-evaluation Monte-Carlo test set.
//
// Plays HastyBot-vs-HastyBot games and, from each game, samples one post-move
// position (data/gcg_post_move.h): a training-eligible turn (the bag had tiles
// when it began -- binary_log.h's eligible_span) whose move placed tiles, so
// the recorded final move is a PLAY the way the truth and the encoder read it.
// The game is truncated right after that move and written as a GCG from the
// final mover's POV -- exactly the input `monte_carlo_sim_tool` scores.
//
// The emitted GCG keeps each move line's rack_before (the sim reads the final
// mover's leave, and the opponent's retained leave, from them) but writes NO
// #Rack pragmas and no end-of-game rack adjustments, so neither player's
// post-move draw ever appears.
//
// Output: positions/<lexicon>/<dataset-name>/part-NNN.gcgs, each a concatenation
// of up to --per-file GCG blocks. Every block begins with `#character-encoding`,
// the record boundary the Python wrapper splits on before scoring.

#include "agent/macondo_bot.h"
#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "game/game.h"
#include "game/move.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {
namespace {

namespace fs = std::filesystem;

// The turns a harvested position may end on: training-eligible and a tile
// placement.
std::vector<int> qualifying_turns(const GameLogStorage& log) {
  const binlog::EligibleSpan span = binlog::eligible_span(log.view());
  std::vector<int> out;
  for (int i = span.begin; i < span.end; ++i)
    if (log.turns[i].move.type() == MoveType::PLAY) out.push_back(i);
  return out;
}

// Truncate the log to end right after turn `last`, and set final_scores to the
// last kept cumulative so the writer emits no END_RACK adjustment lines.
GameLogStorage truncate_after(GameLogStorage log, int last) {
  log.turns.resize(last + 1);
  log.final_scores = log.turns.back().cumulative_scores;
  log.final_racks = {};
  log.end_reason.clear();
  return log;
}

// The GCG for one harvested position: rack_before kept per line, no #Rack pragmas.
std::string harvested_gcg(GameLogStorage log, int last, uint64_t seed, const std::string& lexicon) {
  const GameLogStorage truncated = truncate_after(std::move(log), last);
  GcgWriteOptions options;
  options.lexicon_name = lexicon;
  options.notes = {std::format("Harvested post-move position (game seed {})", seed)};
  return game_log_to_gcg(truncated.view(), options);
}

// Play game `seed` and, if it has any qualifying turn, return the GCG for one
// sampled uniformly (RNG seeded by the game, for reproducibility). Empty if none.
std::string harvest_from_game(HastyBotAgent& a0, HastyBotAgent& a1, const Dictionary& dict,
                              uint64_t seed, const std::string& lexicon) {
  Game game(a0, a1, dict, seed);
  game.play();
  GameLogStorage log = game.extract_log();
  const std::vector<int> turns = qualifying_turns(log);
  if (turns.empty()) return {};
  std::mt19937_64 rng(seed);
  const int pick = turns[std::uniform_int_distribution<std::size_t>(0, turns.size() - 1)(rng)];
  return harvested_gcg(std::move(log), pick, seed, lexicon);
}

std::string part_filename(int part) { return std::format("part-{:03}.gcgs", part); }

// Write the collected GCG blocks into per_file-sized bundle files.
void write_bundles(const fs::path& dir, const std::vector<std::string>& gcgs, int per_file) {
  const int num_parts = (int(gcgs.size()) + per_file - 1) / per_file;
  for (int part = 0; part < num_parts; ++part) {
    std::ofstream os(dir / part_filename(part));
    const int end = std::min(int(gcgs.size()), (part + 1) * per_file);
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

    po::options_description desc("harvest_positions_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "dataset-name", po::value<std::string>(&dataset_name)->default_value(dataset_name),
      "dataset under positions/<lexicon>/; bundle files are written here")(
      "count", po::value<int>(&count)->default_value(count), "number of positions to harvest")(
      "per-file", po::value<int>(&per_file)->default_value(per_file), "GCG blocks per bundle file")(
      "seed", po::value<long>(&base_seed)->default_value(base_seed),
      "base game seed (scans base_seed, base_seed+1, ...; a reserved range disjoint from "
      "training)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::load_dictionary_or_throw();
    const std::string& lexicon = scribblez::Lexicon::instance().name();
    scribblez::HastyEquity::ensure_initialized(lexicon);
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
    while (int(gcgs.size()) < count) {
      std::string gcg = scribblez::harvest_from_game(a0, a1, dict, uint64_t(seed), lexicon);
      if (!gcg.empty()) gcgs.push_back(std::move(gcg));
      ++seed;
      ++scanned;
    }
    scribblez::write_bundles(dir, gcgs, per_file);

    std::cerr << "Harvested " << gcgs.size() << " positions from " << scanned << " games into "
              << dir.string() << " (" << ((count + per_file - 1) / per_file) << " bundle files)\n";
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
