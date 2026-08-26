// Offline generator of evidence trajectories (.sobs sidecars stamped
// kSimObsFlagTrajectory) -- the training data of docs/sim_residual_feedback.md's
// evidence conditioning and proves-best head (docs/roadmap.md, item 4). The
// per-position recipe (anchor, student-model proposals, uniform tail, all under
// common random numbers) is training/evidence_trajectory.h; this tool supplies
// the decision points from one of two front-ends:
//
// - .slog self-play data (--slog-dir / --slog-file): a sampled subset of each
//   game's training-eligible turns, replayed to the pre-move decision point.
//   One .sobs per .slog, positions keyed (game, turn) as the .mset labeling
//   keys them (the two tools share the seed stream, see data/slog_sampling.h).
// - .gcg position sets (--gcg / --gcg-dir, into --out-dir): each file's final
//   recorded state, the side to move holding its #RackN pragma rack
//   (read_gcg_position). One .sobs per .gcg, its single position keyed
//   (0, decision turn). This is how the hand-maintained sets under positions/
//   get their trajectory sidecars.
//
// The student model is required (--model): generation-0 equity-top-K evidence
// is sim_obs_tool's job. Inference runs on a single dedicated thread
// (NeuralNet's one-net-one-thread contract) that position workers round-trip
// through; the sims dominate wall-clock, so the serialization is free.

#include "data/binary_log.h"
#include "data/gcg_reader.h"
#include "data/sim_observation_log.h"
#include "data/slog_sampling.h"
#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_util.h"
#include "training/evidence_trajectory.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;
using binlog::FileHeader;
using binlog::GameMetadata;
using binlog::GamePositionIndex;
using evidence::DecisionPoint;
using evidence::StudentScorer;
using evidence::StudentService;
using evidence::TrajectoryResult;
using evidence::TrajectoryRunner;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string gcg_dir;
  std::vector<std::string> gcg_files;
  std::string out_dir;  // gcg mode: where the .sobs go
  bool open_leaves = false;
  std::string leaf_model;  // value truncation; required with, and only with, --horizon
  evidence::TrajectoryOptions traj;
  int positions_per_game = 1;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file (a cap makes smoke runs cheap)

  bool gcg_mode() const { return !gcg_dir.empty() || !gcg_files.empty(); }
};

// Reject an unusable invocation before any input is read and before any
// worker thread exists (a runner-constructor throw inside a worker would
// terminate the process). --positions-per-game is rejected at 0 for the same
// reason as sim_obs_tool: an empty .sobs would stand in for real evidence.
void validate(const Options& opt) {
  evidence::validate(opt.traj);
  if ((opt.traj.horizon > 0) != !opt.leaf_model.empty()) {
    throw util::CleanException("--horizon and --leaf-model come together");
  }
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  const bool slog_mode = !opt.slog_dir.empty() || !opt.slog_files.empty();
  if (slog_mode == opt.gcg_mode()) {
    throw util::CleanException(
      "give either .slog inputs (--slog-dir/--slog-file) or .gcg inputs (--gcg-dir/--gcg), not "
      "both and not neither");
  }
  if (opt.gcg_mode() && opt.out_dir.empty()) {
    throw util::CleanException("--out-dir is required with .gcg inputs");
  }
}

uint32_t file_flags(const Options& opt) {
  return kSimObsFlagTrajectory | (opt.open_leaves ? kSimObsFlagOpenLeaves : 0u);
}

// What every worker shares: the lexicon and encoding, the options, the
// scorer over the loaded proposer model, and -- under --horizon -- the
// truncation leaf service (shared freely: EvalService serializes its
// callers) and its content hash.
struct Shared {
  const Dictionary& dict;
  const InputEncodingSpec& spec;
  const Options& opt;
  StudentScorer* scorer;
  nn::PositionEvalService* leaf_eval_service;  // null = terminal rollouts
  const std::string& leaf_hash;
};

void add_result(SimObsWriter* writer, const Options& opt, uint32_t game_idx, uint32_t turn_idx,
                uint64_t base_seed, const TrajectoryResult& t) {
  writer->add_position(game_idx, turn_idx, t.candidates, t.observations,
                       uint32_t(opt.traj.rollouts), base_seed, t.num_legal_moves,
                       t.uniform_tail ? kSimObsPosFlagUniformTail : 0u);
}

// Runs a front-end's work items across opt.threads worker threads plus the
// scorer's own thread. A front-end declares its `Item` type and a `Worker`
// constructible from the front-end, with `run(index, item, runner)`; each
// thread owns one Worker and one TrajectoryRunner and claims items off a
// shared index. Results are written by index, so the output is canonically
// ordered and byte-stable across thread counts.
template <typename Front>
void worker_thread(const Shared& sh, const Front& front, std::atomic<size_t>* next,
                   util::ProgressMeter* meter) {
  typename Front::Worker worker(front);
  TrajectoryRunner runner(sh.dict, sh.spec, sh.opt.traj, sh.scorer, sh.leaf_eval_service);
  const std::vector<typename Front::Item>& work = front.work;
  for (size_t i = next->fetch_add(1); i < work.size(); i = next->fetch_add(1)) {
    worker.run(i, work[i], runner);
    meter->add_done();
  }
}

template <typename Front>
void run_positions(const Shared& sh, const Front& front, util::ProgressMeter* meter) {
  std::atomic<size_t> next{0};
  std::thread gpu(&StudentScorer::run, sh.scorer);
  std::vector<std::thread> workers;
  const int threads = std::clamp<int>(sh.opt.threads, 1, std::max<size_t>(1, front.work.size()));
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back(worker_thread<Front>, std::cref(sh), std::cref(front), &next, meter);
  }
  for (auto& w : workers) w.join();
  sh.scorer->stop();
  gpu.join();
}

// --- the .slog front-end ---

// A completed position: what SimObsWriter::add_position consumes.
struct SlogResult {
  uint64_t base_seed;
  TrajectoryResult traj;
};

struct SlogFront {
  using Item = GamePositionIndex;

  const char* buf;
  const InputEncodingSpec& spec;
  const Options& opt;
  std::vector<Item> work;
  std::vector<SlogResult>* results;

  // Replay to the pre-move decision point, then the recipe. Owns its replay
  // scratch and encoder.
  class Worker {
   public:
    explicit Worker(const SlogFront& front) : front_(front), encoder_(front.spec) {}
    void run(size_t i, const GamePositionIndex& w, TrajectoryRunner& runner);

   private:
    const SlogFront& front_;
    std::vector<TurnRecord> scratch_;
    binlog::PositionEncoder encoder_;
  };
};

void SlogFront::Worker::run(size_t i, const GamePositionIndex& w, TrajectoryRunner& runner) {
  const Options& opt = front_.opt;
  const GameLog g = binlog::make_game_view(front_.buf, w.game_idx, scratch_, nullptr);
  const int mover = encoder_.replay_to_sampled(g, int(w.turn_idx), /*post_move=*/false);
  DecisionPoint dp;
  dp.pos.board = encoder_.enc().board();
  dp.pos.scores = {encoder_.enc().score(0), encoder_.enc().score(1)};
  dp.pos.mover = mover;
  dp.pos.rack = encoder_.rack(mover);
  if (opt.open_leaves) {
    dp.pos.opp_leave = binlog::opp_leave_from_replay(g, int(w.turn_idx), encoder_.rack(1 - mover));
  }
  dp.enc = &encoder_.enc();
  dp.bag_size = encoder_.bag_size();
  SlogResult& out = (*front_.results)[i];
  out.base_seed = binlog::position_seed(opt.seed, w.game_idx, w.turn_idx);
  out.traj = runner.run(dp, out.base_seed);
}

// Generate the trajectory .sobs sidecar for one loaded .slog file.
void process_slog(const Shared& sh, const std::vector<char>& buf, const fs::path& sobs_path,
                  const std::string& proposer_hash, util::ProgressMeter* meter) {
  const Options& opt = sh.opt;
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));

  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<SlogResult> results;
  SlogFront front{buf.data(), sh.spec, opt, {}, &results};
  for (uint32_t g = 0; g < num_games; ++g) {
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &front.work);
  }
  std::sort(front.work.begin(), front.work.end());
  results.resize(front.work.size());
  run_positions(sh, front, meter);

  SimObsWriter writer(sobs_path.string(), file_flags(opt), proposer_hash, sh.leaf_hash,
                      opt.traj.horizon);
  for (size_t i = 0; i < front.work.size(); ++i) {
    add_result(&writer, opt, front.work[i].game_idx, front.work[i].turn_idx, results[i].base_seed,
               results[i].traj);
  }
  writer.close();
}

void run_slog_mode(const Dictionary& dict, const InputEncodingSpec& spec, const Options& opt,
                   StudentService* service, const std::string& proposer_hash,
                   nn::PositionEvalService* leaf_eval_service, const std::string& leaf_hash) {
  // Games played face up must be simmed face up (see sim_obs_tool for why
  // the reverse pairing is allowed).
  const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
    binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), ".sobs", opt.open_leaves,
    "{} was played with face-up leaves; pass --open-leaves to sim it");
  if (pending.empty()) return;
  uint64_t total_positions = 0;
  for (const binlog::PendingSlog& p : pending)
    total_positions +=
      binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
  std::cerr << "evidence trajectories: " << pending.size() << " file(s), " << total_positions
            << " positions; anchor + " << opt.traj.proposals_min << ".." << opt.traj.proposals_max
            << " proposals + uniform tail x " << opt.traj.rollouts << " rollouts, proposer "
            << proposer_hash.substr(0, 12) << ", " << opt.threads << " threads\n";

  util::ProgressMeter meter(total_positions, "positions");
  for (const binlog::PendingSlog& p : pending) {
    StudentScorer scorer(service);
    const Shared sh{dict, spec, opt, &scorer, leaf_eval_service, leaf_hash};
    process_slog(sh, p.bytes, p.sidecar(".sobs"), proposer_hash, &meter);
  }
  meter.finish("evidence trajectories");
}

// --- the .gcg front-end ---

std::vector<fs::path> resolve_gcg_inputs(const Options& opt) {
  std::vector<fs::path> files;
  for (const std::string& f : opt.gcg_files) files.emplace_back(f);
  if (!opt.gcg_dir.empty()) {
    for (const auto& e : fs::directory_iterator(opt.gcg_dir)) {
      if (e.path().extension() == ".gcg") files.push_back(e.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

fs::path gcg_sobs_path(const Options& opt, const fs::path& gcg) {
  return fs::path(opt.out_dir) / gcg.filename().replace_extension(".sobs");
}

std::string read_text(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw util::CleanException("cannot read {}", path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct GcgWork {
  fs::path path;
  ParsedGcgPosition position;
};

struct GcgResult {
  uint64_t base_seed;
  TrajectoryResult traj;
};

struct GcgFront {
  using Item = GcgWork;

  const InputEncodingSpec& spec;
  const Options& opt;
  std::vector<Item> work;
  std::vector<GcgResult>* results;

  // Replay the parsed game to its decision point, then the recipe.
  class Worker {
   public:
    explicit Worker(const GcgFront& front) : front_(front) {}
    void run(size_t i, const GcgWork& w, TrajectoryRunner& runner);

   private:
    const GcgFront& front_;
  };
};

void GcgFront::Worker::run(size_t i, const GcgWork& w, TrajectoryRunner& runner) {
  const ParsedGcgPosition& p = w.position;
  // Replay the recorded moves into a fresh encoder. apply_move attributes
  // each to the encoder's own turn order (seat 0 first), which the recorded
  // seats must follow for scores and last moves to land on the right player.
  GameStateEncoder enc(front_.spec);
  for (const ParsedGcgTurn& t : p.game.turns) enc.apply_move(t.record.move);
  RELEASE_ASSERT(enc.active_player() == p.mover);
  DecisionPoint dp;
  dp.pos.board = p.board;
  dp.pos.scores = p.scores;
  dp.pos.mover = p.mover;
  dp.pos.rack = p.rack;
  dp.pos.opp_leave = p.opp_leave;
  dp.enc = &enc;
  dp.bag_size = p.bag_size;
  GcgResult& out = (*front_.results)[i];
  out.base_seed = binlog::position_seed(front_.opt.seed, 0, uint32_t(p.turns));
  out.traj = runner.run(dp, out.base_seed);
}

// Parse every pending .gcg up front, so a malformed file fails the run before
// any sim is spent, and an endgame position (which the sims cannot run) is
// named rather than crashing a worker.
std::vector<GcgWork> load_pending_gcgs(const Options& opt) {
  std::vector<GcgWork> work;
  for (const fs::path& gcg : resolve_gcg_inputs(opt)) {
    if (fs::exists(gcg_sobs_path(opt, gcg))) continue;
    GcgWork item{gcg, {}};
    std::string error;
    if (!read_gcg_position(read_text(gcg), opt.open_leaves, &item.position, &error)) {
      throw util::CleanException("{}: {}", gcg.string(), error);
    }
    if (item.position.bag_size <= 0) {
      throw util::CleanException(
        "{}: the bag is empty at the decision point; the sims need a non-empty bag", gcg.string());
    }
    work.push_back(std::move(item));
  }
  return work;
}

void run_gcg_mode(const Dictionary& dict, const InputEncodingSpec& spec, const Options& opt,
                  StudentService* service, const std::string& proposer_hash,
                  nn::PositionEvalService* leaf_eval_service, const std::string& leaf_hash) {
  std::vector<GcgResult> results;
  GcgFront front{spec, opt, load_pending_gcgs(opt), &results};
  if (front.work.empty()) return;
  results.resize(front.work.size());
  fs::create_directories(opt.out_dir);
  std::cerr << "evidence trajectories: " << front.work.size() << " gcg position(s); anchor + "
            << opt.traj.proposals_min << ".." << opt.traj.proposals_max
            << " proposals + uniform tail x " << opt.traj.rollouts << " rollouts, proposer "
            << proposer_hash.substr(0, 12) << ", " << opt.threads << " threads\n";

  util::ProgressMeter meter(front.work.size(), "positions");
  StudentScorer scorer(service);
  const Shared sh{dict, spec, opt, &scorer, leaf_eval_service, leaf_hash};
  run_positions(sh, front, &meter);
  meter.finish("evidence trajectories");

  for (size_t i = 0; i < front.work.size(); ++i) {
    const GcgWork& w = front.work[i];
    SimObsWriter writer(gcg_sobs_path(opt, w.path).string(), file_flags(opt), proposer_hash,
                        leaf_hash, opt.traj.horizon);
    add_result(&writer, opt, 0, uint32_t(w.position.turns), results[i].base_seed, results[i].traj);
    writer.close();
  }
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    evidence::TrajectoryOptions& traj = opt.traj;
    nn::NeuralNetParams<nn::MoveSetEvaluationSpec> params;
    po::options_description desc("evidence_trajectory_generator options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .sobs sidecar gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "gcg-dir", po::value<std::string>(&opt.gcg_dir),
      "directory of .gcg position files; each without a .sobs in --out-dir gets one")(
      "gcg", po::value<std::vector<std::string>>(&opt.gcg_files),
      "explicit .gcg position file to process (repeatable; adds to --gcg-dir)")(
      "out-dir", po::value<std::string>(&opt.out_dir),
      "where .gcg inputs' .sobs go, named <gcg stem>.sobs (required with .gcg inputs)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim and score with the opponent's retained leave known -- required for face-up-leaves "
      "games, and must match the model's input arm")(
      "rollouts", po::value<int>(&traj.rollouts)->default_value(traj.rollouts),
      "Monte-Carlo rollouts per candidate")(
      "horizon", po::value<int>(&traj.horizon)->default_value(traj.horizon),
      "value truncation: rollouts stop after this many plies and --leaf-model scores the "
      "horizon; 0 rolls out to a natural game end")(
      "leaf-model", po::value<std::string>(&opt.leaf_model),
      "position evaluation model (.onnx) scoring rollout horizons; required with, and only "
      "with, --horizon")(
      "proposals-min", po::value<int>(&traj.proposals_min)->default_value(traj.proposals_min),
      "least model proposals per trajectory (the randomized length's lower bound)")(
      "proposals-max", po::value<int>(&traj.proposals_max)->default_value(traj.proposals_max),
      "most model proposals per trajectory")(
      "temperature", po::value<double>(&traj.temperature)->default_value(traj.temperature),
      "softmax temperature over the model's win-equity scores (win-equity units)")(
      "proposal-pool", po::value<int>(&traj.proposal_pool)->default_value(traj.proposal_pool),
      "proposals are drawn from the model's top-N unsimmed candidates")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game (.slog inputs)")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed; with .slog inputs it MUST match the target generator's --seed for its sampled "
      "positions to contain this tool's (the forced-candidate labeling relies on it)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each .slog (0 = all); for smoke runs");
    params.add_options(desc);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    StudentService service(params);
    service.load();
    if (service.opp_leave_input() != opt.open_leaves) {
      throw util::CleanException(
        "the model's input arm (opp_leave_input={}) must match --open-leaves={}: a mismatched "
        "proposer would score moves under an information condition the sims do not share",
        service.opp_leave_input(), opt.open_leaves);
    }
    const InputEncodingSpec spec{&dict, service.opp_leave_input()};
    const std::string proposer_hash = nn::content_hash(binlog::read_file_bytes(params.onnx_path));

    // The truncation leaf service, shared by every position worker (the
    // runners are single-threaded, but many run at once; EvalService
    // serializes their calls).
    std::unique_ptr<nn::PositionEvalService> leaf_eval_service;
    std::string leaf_hash;
    if (!opt.leaf_model.empty()) {
      nn::NeuralNetParams<nn::PositionEvaluationSpec> leaf_params;
      leaf_params.onnx_path = opt.leaf_model;
      leaf_params.cuda_device_id = params.cuda_device_id;
      leaf_eval_service = nn::make_loaded_service(leaf_params);
      leaf_hash = nn::content_hash(binlog::read_file_bytes(opt.leaf_model));
    }

    if (opt.gcg_mode()) {
      run_gcg_mode(dict, spec, opt, &service, proposer_hash, leaf_eval_service.get(), leaf_hash);
    } else {
      run_slog_mode(dict, spec, opt, &service, proposer_hash, leaf_eval_service.get(), leaf_hash);
    }
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
