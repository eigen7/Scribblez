// harvest_positions_tool: harvests positions for the large position-evaluation
// test set. py/scripts/build_position_eval_test_set.py drives it, splits the
// bundles into one GCG per position, and scores them with monte_carlo_sim_tool.
//
//   harvest_positions_tool --dataset-name position-eval-test-dataset-large --count 1000
//
// It plays HastyBot-vs-HastyBot games with consecutive seeds and samples one
// post-move position (data/gcg_post_move.h) from each game. The sampled turn
// must be training-eligible (binary_log.h's eligible_span: the bag had tiles
// when the turn began) and must place tiles, since the ground truth and the
// encoder both read the last move as a play. The game is cut off right after
// that move and written as a GCG.
//
// Each move line keeps its rack_before, because the sim reads both players'
// leaves from those. The GCG has no #Rack pragmas and no end-of-game rack
// lines, so neither player's post-move draw is revealed.
//
// Output goes to positions/<lexicon>/<dataset-name>/part-NNN.gcgs, relative to
// the working directory. Each bundle concatenates up to --per-file GCG blocks,
// and every block begins with `#character-encoding`, the record boundary the
// Python driver splits on.

#include "agent/hasty_bot.h"
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

// Truncate the log to end right after turn `last`. Setting final_scores to the
// last cumulative scores keeps the writer from emitting end-of-game rack lines.
GameLogStorage truncate_after(GameLogStorage log, int last) {
  log.turns.resize(last + 1);
  log.final_scores = log.turns.back().cumulative_scores;
  log.final_racks = {};
  log.end_reason.clear();
  return log;
}

std::string harvested_gcg(GameLogStorage log, int last, uint64_t seed, const std::string& lexicon) {
  const GameLogStorage truncated = truncate_after(std::move(log), last);
  GcgWriteOptions options;
  options.lexicon_name = lexicon;
  options.notes = {std::format("Harvested post-move position (game seed {})", seed)};
  return game_log_to_gcg(truncated.view(), options);
}

// Play game `seed` and return the GCG for one uniformly sampled qualifying turn,
// or an empty string if there is none. The sample is seeded by the game seed,
// so a harvest is reproducible.
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
      "first game seed; games use base_seed, base_seed+1, ... (keep this range disjoint from "
      "the seeds used for training games)");
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
