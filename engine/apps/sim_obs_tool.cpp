// Offline generator of Monte-Carlo sim observations (.sobs sidecars) for
// .slog self-play data -- the sim-evidence inputs of
// docs/plans/sim_residual_feedback.md and the data source for its kill-test.
//
// For a sampled subset of each game's training-eligible turns, the tool
// replays the game to the pre-move decision point, ranks the legal candidates
// (plays and exchanges) by HastyBot static equity, and runs SimRunner over the
// top-K (common random numbers, HastyBot rollouts to a natural end). Each
// processed .slog file gets a same-stem .sobs sidecar. The input is either
// --slog-dir (every .slog in the directory; files whose sidecar already
// exists are skipped, so an interrupted run resumes by rerunning) or one or
// more explicit --slog-file arguments (what generate_kill_test_data.py
// passes, having already selected the files missing a sidecar). Progress
// across all positions of the invocation renders as a bar on stderr (TTY
// only), and a timing summary prints at the end.
//
// Because the self-play games are HastyBot's own, the equity-argmax candidate
// (rank 0) is the move that was actually played at the position, so each
// position's evidence set contains the played move's own sim.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "data/sim_observation_log.h"
#include "data/slog_sampling.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "sim/slog_position_simmer.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;
using binlog::FileHeader;
using binlog::GameMetadata;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  bool open_leaves = false;
  int rollouts = 200;
  int horizon = 0;         // value truncation; 0 = terminal rollouts
  std::string leaf_model;  // required with, and only with, --horizon
  int top_k = 10;
  int positions_per_game = 1;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)
};

// The shared simmer's config for `opt`, over the shared truncation leaf service
// (EvalService serializes its callers) -- null for terminal rollouts.
SlogSimConfig sim_config(const Options& opt, nn::PositionEvalService* leaf_eval_service) {
  SlogSimConfig c;
  c.open_leaves = opt.open_leaves;
  c.selector = recipe_selector({.top_k = opt.top_k, .quotas = {}});
  c.runner.rollouts = opt.rollouts;
  c.runner.threads = 1;
  c.runner.horizon_plies = opt.horizon;
  c.runner.leaf_service = leaf_eval_service;
  c.seed = opt.seed;
  c.threads = opt.threads;
  return c;
}

// Reject an unusable invocation before a single .slog is read -- and, for the
// SimRunner params, before any worker thread exists: the runners are built
// inside the workers, where a constructor throw would escape the thread and
// terminate the process instead of printing an error.
//
// Both caps are rejected at 0 rather than read as "no cap". A run that samples
// no positions, or ranks no candidates, still writes a .sobs sidecar for every
// input file, and a sidecar's existence is what makes later runs skip its
// .slog -- so the empty output would silently stand in for the real evidence
// until someone noticed the corpus was hollow.
void validate(const Options& opt) {
  SimRunner::validate_horizon("sim-obs-tool", opt.horizon, !opt.leaf_model.empty());
  Options terminal = opt;  // the leaf service does not exist yet
  terminal.horizon = 0;
  SimRunner::validate(sim_config(terminal, nullptr).runner);
  if (opt.top_k < 1) throw util::CleanException("--top-k must be >= 1");
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
}

using binlog::GamePositionIndex;

// Generate the .sobs sidecar for one loaded .slog file.
void process_file(const std::vector<char>& buf, const fs::path& sobs_path, const Dictionary& dict,
                  const Options& opt, nn::PositionEvalService* leaf_eval_service,
                  const std::string& leaf_hash, util::ProgressMeter* meter) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));

  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g)
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &work);
  std::sort(work.begin(), work.end());

  // Prepend the file, so a batch run's failure names both file and position.
  std::vector<SimmedPosition> results;
  try {
    results = sim_slog_positions(buf, dict, sim_config(opt, leaf_eval_service), work, meter);
  } catch (const util::Exception& ex) {
    throw util::Exception("{}: {}", sobs_path.stem().string(), ex.what());
  }

  // The work list is sorted by (game, turn) and results are indexed by work
  // slot, so the output is canonically ordered and byte-stable across thread
  // counts.
  SimObsWriter writer(sobs_path.string(), opt.open_leaves ? kSimObsFlagOpenLeaves : 0,
                      /*proposer_hash=*/{}, leaf_hash, opt.horizon);
  for (const SimmedPosition& r : results) {
    writer.add_position(r.pos.game_idx, r.pos.turn_idx, r.candidates.moves, r.observations,
                        uint32_t(opt.rollouts), r.base_seed, r.candidates.num_legal_moves);
  }
  writer.close();
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("sim_obs_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .sobs sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim with the opponent's retained leave known (their replenishment draws stay "
      "hidden and sampled) -- the open-leaves information condition; recorded in the "
      ".sobs header flags")("rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
                            "Monte-Carlo rollouts per candidate")(
      "horizon", po::value<int>(&opt.horizon)->default_value(opt.horizon),
      "value truncation: rollouts stop after this many plies and --leaf-model scores the "
      "horizon; 0 rolls out to a natural game end")(
      "leaf-model", po::value<std::string>(&opt.leaf_model),
      "position evaluation model (.onnx) scoring rollout horizons; required with, and only "
      "with, --horizon")("top-k", po::value<int>(&opt.top_k)->default_value(opt.top_k),
                         "candidates simmed per position (HastyBot-equity ranked)")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed (drives position sampling and rollout seeds)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all); for smoke runs");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    // Games played face up must be simmed face up. The reverse is fine and
    // deliberate: open-leaves sims over a standard corpus are the
    // information-condition instrument (docs/plans/sim_residual_feedback.md), which
    // hands the sims more than the players had. Sims that know LESS than the
    // players did are the incoherent direction -- the evidence would describe
    // a game nobody played.
    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), ".sobs", opt.open_leaves,
      "{} was played with face-up leaves; pass --open-leaves to sim it");
    if (pending.empty()) return 0;
    uint64_t total_positions = 0;
    for (const binlog::PendingSlog& p : pending)
      total_positions +=
        binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
    std::cerr << "sim-obs: " << pending.size() << " file(s), " << total_positions << " positions; "
              << opt.top_k << " candidates x " << opt.rollouts << " rollouts, " << opt.threads
              << " threads\n";

    // The truncation leaf service, shared by every position worker (the
    // runners are single-threaded, but many run at once; EvalService
    // serializes their calls).
    std::shared_ptr<nn::PositionEvalService> leaf_eval_service =
      nn::load_leaf_position_service(opt.leaf_model);
    std::string leaf_hash;
    if (!opt.leaf_model.empty())
      leaf_hash = nn::content_hash(binlog::read_file_bytes(opt.leaf_model));

    util::ProgressMeter meter(total_positions, "positions");
    for (const binlog::PendingSlog& p : pending) {
      process_file(p.bytes, p.sidecar(".sobs"), dict, opt, leaf_eval_service.get(), leaf_hash,
                   &meter);
    }
    meter.finish("sim-obs");
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
