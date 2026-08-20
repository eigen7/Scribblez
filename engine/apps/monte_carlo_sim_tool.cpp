// Offline Monte-Carlo ground-truth generator for the position-evaluation test
// datasets.
//
// Each input GCG is a post-move position (data/gcg_post_move.h): the final
// recorded move is the evaluated player's, the other player acts next. From
// the final player's POV, this plays the position out many times --
// EndgameHastyBot vs EndgameHastyBot, sampling the hidden tiles and bag -- and
// records the exact win/loss/draw and final-score-delta distribution. The
// aggregate converges to the true value the position evaluation model should
// predict.
//
// The truth depends on what a rollout knows of the opponent's leave, so it is
// computed under both information conditions (sim/monte_carlo_sim.h) and
// written as <dataset>/monte-carlo-sim-results.<condition>.json, keyed by GCG
// stem (e.g. "pos-7"). A position whose opponent kept nothing (a bingo) has
// one truth, rolled once and written under both.

#include "belief/rack_inference.h"
#include "data/gcg_post_move.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/monte_carlo_sim.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/misc.h"
#include "util/string.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace json = boost::json;

constexpr scribblez::LeaveCondition kConditions[] = {scribblez::LeaveCondition::kHidden,
                                                     scribblez::LeaveCondition::kFaceUp};

fs::path results_path(const fs::path& dataset, scribblez::LeaveCondition condition) {
  return dataset /
         std::format("monte-carlo-sim-results.{}.json", scribblez::leave_condition_name(condition));
}

// --condition: which of kConditions to score, as include flags.
std::array<bool, 2> parse_conditions(const std::string& arg) {
  if (arg == "both") return {true, true};
  for (int c = 0; c < 2; ++c) {
    if (arg == scribblez::leave_condition_name(kConditions[c])) return {c == 0, c == 1};
  }
  throw scribblez::util::CleanException(
    "--condition must be 'both', 'hidden-leaves', or 'face-up-leaves' (got '{}')", arg);
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string dataset_name = "position-eval-test-dataset";
    int games = 10000;
    int threads = scribblez::util::default_thread_count();
    scribblez::belief::RackInferrer::Params infer;
    std::string condition = "both";

    po::options_description desc("monte_carlo_sim_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "dataset-name", po::value<std::string>(&dataset_name)->default_value(dataset_name),
      "dataset under positions/<lexicon>/; the results JSON is written here")(
      "games", po::value<int>(&games)->default_value(games),
      "Monte-Carlo rollouts per position (seeds 1..games)")(
      "threads", po::value<int>(&threads)->default_value(threads), "parallel workers")(
      "condition", po::value<std::string>(&condition)->default_value(condition),
      "which information condition to score: both, hidden-leaves, or face-up-leaves "
      "(one condition regenerates just its results file)")(
      "infer-temperature", po::value<double>(&infer.temperature)->default_value(infer.temperature),
      "hidden-leaves condition: the leave inference's likelihood temperature (equity points)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);
    const std::array<bool, 2> selected = parse_conditions(condition);

    const scribblez::Dictionary& dict = scribblez::load_dictionary_or_throw();
    scribblez::HastyEquity::ensure_initialized(scribblez::Lexicon::instance().name());
    const std::string& lexicon = scribblez::Lexicon::instance().name();
    const fs::path dataset = fs::path("positions") / lexicon / dataset_name;
    std::cerr << "Lexicon: " << lexicon << "; dataset: " << dataset.string() << "; " << games
              << " rollouts/position on " << threads << " threads\n";

    std::vector<fs::path> gcgs;
    for (const auto& entry : fs::directory_iterator(dataset))
      if (entry.path().extension() == ".gcg") gcgs.push_back(entry.path());
    std::sort(gcgs.begin(), gcgs.end(), scribblez::util::path_natural_less);

    std::array<json::object, 2> out;  // indexed like kConditions
    for (const fs::path& gcg : gcgs) {
      scribblez::ParsedGcgPostMove pos;
      std::string err;
      if (!scribblez::read_gcg_post_move(scribblez::util::read_file(gcg.string()), &pos, &err)) {
        std::cerr << "  SKIP " << gcg.filename().string() << ": " << err << "\n";
        continue;
      }
      const std::string stem = gcg.stem().string();
      for (int c = 0; c < 2; ++c) {
        if (!selected[c]) continue;
        // An empty opponent leave makes the conditions coincide: nothing is
        // known face-up, and hidden has nothing to infer from. Reuse the
        // hidden result when this run computed it.
        if (c == 1 && selected[0] && pos.opp_leave.empty()) {
          out[1][stem] = out[0][stem];
          continue;
        }
        const scribblez::MonteCarloResult r =
          scribblez::run_monte_carlo(pos, dict, games, threads, kConditions[c], infer);
        out[c][stem] = r.to_json();
        std::cerr << "  " << stem << " [" << scribblez::leave_condition_name(kConditions[c])
                  << "]: W/L/D = " << r.wins << "/" << r.losses << "/" << r.draws << " over "
                  << r.delta_hist.size() << " distinct deltas\n";
      }
    }

    for (int c = 0; c < 2; ++c) {
      if (!selected[c]) continue;
      const fs::path out_path = results_path(dataset, kConditions[c]);
      // Unlink before writing: were the destination a symlink (one condition's
      // file once aliased the other's), an ofstream would write THROUGH it and
      // clobber the file it points at.
      fs::remove(out_path);
      std::ofstream os(out_path);
      scribblez::util::pretty_print(os, json::value(std::move(out[c])));
      os << "\n";
      std::cerr << "Wrote " << out_path.string() << "\n";
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
