// belief_sidecar_tool: writes the .belief sidecar (data/belief_log.h) beside
// each .slog, the mover's belief about the opponent's leave at every turn of
// the leave inference rows' region, under the model `--leave-inference`
// names. Each .slog's .forgone sidecar must be in place (forgone_play_tool).
// The deep_bot_leaf_eval trainer runs it over each generation and its
// held-out games (py/scribblez/deep_bot_leaf_eval/beliefs.py).
//
//   belief_sidecar_tool --slog-dir gen/ --leave-inference leave.onnx --threads 16
//   belief_sidecar_tool --slog-file a.slog --leave-inference leave.onnx --draws 16
//
// Every input gets a fresh sidecar, replacing any already there. The model
// runs through TensorRT, so the tool needs a GPU.

#include "belief/served_leave_model.h"
#include "data/belief_log.h"
#include "data/binary_log.h"
#include "data/slog_sampling.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/misc.h"
#include "util/work_pool.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace scribblez;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string leave_inference;
  int draws = 0;
  int threads = util::default_thread_count();
};

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("belief_sidecar_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each gets a .belief sidecar")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "leave-inference", po::value<std::string>(&opt.leave_inference)->required(),
      "the leave inference model (ONNX) the beliefs are under")(
      "draws", po::value<int>(&opt.draws)->default_value(opt.draws),
      "the opponent's leaves stored per turn, drawn as DeepBot draws them (the held-out set's)")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    if (opt.draws < 0) throw util::CleanException("--draws must be >= 0");
    util::set_worker_count(opt.threads);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());
    belief::ServedLeaveTable served(opt.leave_inference, HastyEquity::instance().leave_values());
    const BeliefModel model{served.served.predict_fn(), served.served.identity()};
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t games = 0;
    for (const std::filesystem::path& path :
         binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files)) {
      const std::vector<char> slog = binlog::load_slog_input(path);
      write_belief_sidecar(slog, path.string(), dict, served.table, model, opt.draws,
                           belief_path_of(path.string()));
      games += reinterpret_cast<const binlog::FileHeader*>(slog.data())->num_games;
    }
    const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "belief: " << games << " games in " << secs << " s, " << opt.threads
              << " threads\n";
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
