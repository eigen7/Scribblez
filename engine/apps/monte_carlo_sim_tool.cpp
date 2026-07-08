// Offline Monte-Carlo ground-truth generator for the position-evaluation test dataset.
//
// Each input is a penultimate-bingo GCG: the player to move bingoed on the
// penultimate turn (so its rack is a clean full draw), and the other player made
// one final move. From that final player's POV, this plays the position out many
// times -- HastyBot vs HastyBot, sampling the unknown racks and bag -- and records
// the exact win/loss/draw and final-score-delta distribution. The aggregate
// converges to the true value the position evaluation model should predict.
//
// Output: <dataset>/monte-carlo-sim-results.json, keyed by GCG stem (e.g. "pos-7").

#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "selfplay/game_runner.h"
#include "selfplay/monte_carlo_sim.h"
#include "util/cli.h"
#include "util/hardware.h"
#include "util/io.h"
#include "util/json.h"
#include "util/string.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace json = boost::json;

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string dataset_name = "position-eval-test-dataset";
    int games = 10000;
    int threads = scribblez::util::default_thread_count();

    po::options_description desc("monte_carlo_sim_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "dataset-name", po::value<std::string>(&dataset_name)->default_value(dataset_name),
      "dataset under positions/<lexicon>/; the results JSON is written here")(
      "games", po::value<int>(&games)->default_value(games),
      "Monte-Carlo rollouts per position (seeds 1..games)")(
      "threads", po::value<int>(&threads)->default_value(threads), "parallel workers");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    const std::string& lexicon = scribblez::Lexicon::instance().name();
    const fs::path dataset = fs::path("positions") / lexicon / dataset_name;
    std::cerr << "Lexicon: " << lexicon << "; dataset: " << dataset.string() << "; " << games
              << " rollouts/position on " << threads << " threads\n";

    std::vector<fs::path> gcgs;
    for (const auto& entry : fs::directory_iterator(dataset))
      if (entry.path().extension() == ".gcg") gcgs.push_back(entry.path());
    std::sort(gcgs.begin(), gcgs.end(), scribblez::util::path_natural_less);

    json::object out;
    for (const fs::path& gcg : gcgs) {
      scribblez::MonteCarloPosition pos;
      std::string err;
      if (!scribblez::parse_monte_carlo_position(scribblez::util::read_file(gcg.string()), &pos,
                                                 &err)) {
        std::cerr << "  SKIP " << gcg.filename().string() << ": " << err << "\n";
        continue;
      }
      const scribblez::MonteCarloResult r = scribblez::run_monte_carlo(pos, dict, games, threads);
      out[gcg.stem().string()] = r.to_json();
      std::cerr << "  " << gcg.stem().string() << ": W/L/D = " << r.wins << "/" << r.losses << "/"
                << r.draws << " over " << r.delta_hist.size() << " distinct deltas\n";
    }

    const fs::path out_path = dataset / "monte-carlo-sim-results.json";
    std::ofstream os(out_path);
    scribblez::util::pretty_print(os, json::value(std::move(out)));
    os << "\n";
    std::cerr << "Wrote " << out_path.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
