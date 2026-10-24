// policy_set_pool_tool: the policy-set model's labeled pools
// (docs/plans/deep_bot_v0.md, "Data: labeled pools"), on the GPU.
//
//   policy_set_pool_tool --slog-dir chunk/ --leaf-model leaf_eval_gen77.onnx
//   policy_set_pool_tool --slog-dir chunk/ --leaf-model leaf_eval_gen77.onnx
//     --leave-inference leave_inference_gen1188.onnx
//
// Per .slog, --positions-per-game eligible turns of each game; per turn, the
// whole pool (sim/slog_position_simmer.h pool_selector: every play, blank
// plays pruned at --blank-prune-points, every distinct exchange, the pass),
// each candidate simmed up to --rollouts times, raced in rounds
// (sim/pool_labeler.h). A rollout is the candidate, the opponent's and the
// chooser's HastyBot replies, and the leaf network's value of the end state,
// or the exact outcome when the game ended (gpu/gpu_rollout_values.h). Each
// .slog gets a same-stem .pool (data/pool_log.h); files that have one are
// skipped, after a check that it came from the same sources.
//
// --leave-inference seats each rollout's opponent leave from the leave
// inference model's belief (belief/slog_leave_beliefs.h), one leave per
// rollout index shared by every candidate; each .slog then needs its .forgone
// sidecar (forgone_play_tool). Without it, or where the leave is not
// modelled, the opponent's whole rack is drawn uniformly.
//
// --backend cpu runs the same rollouts on the CPU reference with a synthetic
// leaf (a logistic in the spread), for the test suites, which hide the GPU.
//
// The CPU prepares positions (replay, candidates, beliefs, the candidates
// applied) on a producer thread ahead of the GPU, which the labeler keeps fed
// with several positions' rounds per launch.

#include "belief/opp_rack_summary.h"
#include "belief/served_leave_model.h"
#include "belief/slog_leave_beliefs.h"
#include "data/forgone_play_log.h"
#include "data/pool_log.h"
#include "data/slog_sampling.h"
#include "gpu/gpu_rollout_values.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/lexicon.h"
#include "nn/onnx_metadata.h"
#include "nn/trt_util.h"
#include "sim/hasty_rollout_task.h"
#include "sim/pool_labeler.h"
#include "sim/rollout_values.h"
#include "sim/slog_position_simmer.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;

constexpr const char* kPoolExt = ".pool";
constexpr const char* kSyntheticLeaf = "synthetic";
constexpr const char* kLeafFormatKey = "leaf_eval_format_version";
constexpr const char* kLeaveFormatKey = "leave_inference_format_version";
// Salts a rollout's seed for its leave quantile, so the quantile is not
// correlated with the bag's draws, which the same seed drives (as SimRunner
// does).
constexpr uint64_t kLeaveQuantileSalt = 0x1EAFE5EEDull;
// Positions the producer prepares per batch, ahead of the labeler.
constexpr int kPrepBatch = 16;

// The --leave-inference model and the leave table its beliefs are over.
struct LeaveModel {
  LeaveValueArrays table;
  belief::ServedLeaveModel served;

  LeaveModel(const std::string& onnx_path, const LeaveValues& values)
      : table(values), served(onnx_path, values.content_hash()) {}
};

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string backend = "gpu";
  std::string leaf_model;  // an ONNX path, or "synthetic" with --backend cpu
  std::string leaf_precision = "fp16";
  int leaf_rows = nn::LeafEvalSpec::kDefaultMaxRows;
  std::string leave_inference_path;
  int positions_per_game = 2;
  PoolLabeler::Params labeler;
  int blank_prune_points = 40;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;

  bool gpu() const { return backend == "gpu"; }
};

void validate(const Options& opt) {
  if (opt.backend != "gpu" && opt.backend != "cpu")
    throw util::CleanException("--backend must be gpu or cpu");
  if (opt.leaf_model.empty()) throw util::CleanException("--leaf-model is required");
  if (opt.gpu() && opt.leaf_model == kSyntheticLeaf)
    throw util::CleanException("--leaf-model {} needs --backend cpu", kSyntheticLeaf);
  if (!opt.gpu() && opt.leaf_model != kSyntheticLeaf)
    throw util::CleanException("--backend cpu needs --leaf-model {}", kSyntheticLeaf);
  if (opt.labeler.rollouts < 1) throw util::CleanException("--rollouts must be >= 1");
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  if (opt.blank_prune_points < 0) throw util::CleanException("--blank-prune-points must be >= 0");
  if (opt.labeler.launch_tasks < 1 || opt.labeler.max_in_flight < 1)
    throw util::CleanException("--launch-tasks and --max-in-flight must be >= 1");
}

PoolModel pool_model(const nn::ModelIdentity& id) {
  return {id.hash, id.tag, id.generation, id.format_version};
}

// Where this run's labels come from.
PoolSources pool_sources(const Options& opt, const LeaveModel* leave_model) {
  PoolSources s;
  s.leaf_model = opt.gpu() ? pool_model(nn::read_model_identity(opt.leaf_model, kLeafFormatKey))
                           : PoolModel{"", kSyntheticLeaf, -1, 0};
  if (leave_model) s.leave_model = pool_model(leave_model->served.identity());
  s.rollouts = opt.labeler.rollouts;
  return s;
}

// A finished .pool is skipped, not redone, so it must come from this run's
// sources: pools of mixed sources in one corpus would pass for one set of
// labels.
void check_finished_pools(const std::vector<fs::path>& slogs, const PoolSources& want) {
  for (const fs::path& slog : slogs) {
    const fs::path pool = fs::path(slog).replace_extension(kPoolExt);
    if (!fs::exists(pool)) continue;
    const char* field = PoolSources::of(PoolReader(pool.string()).header()).first_difference(want);
    if (field != nullptr) {
      throw util::CleanException("{} has another {} than this run; move it aside to redo it",
                                 pool.string(), field);
    }
  }
}

// Under --leave-inference every .slog needs the .forgone sidecar the model's
// input reads; say so before any work rather than at the first such file.
void check_forgone_sidecars(const std::vector<fs::path>& slogs) {
  for (const fs::path& slog : slogs) {
    if (!fs::exists(forgone_path_of(slog.string())))
      throw util::CleanException("{} has no .forgone sidecar; run forgone_play_tool over it first",
                                 slog.string());
  }
}

// --positions-per-game eligible turns of each game, in canonical order.
std::vector<binlog::GamePositionIndex> sample_work(const std::vector<char>& buf,
                                                   const Options& opt) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(buf.data() + sizeof(binlog::FileHeader));
  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  std::vector<binlog::GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g)
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &work);
  std::sort(work.begin(), work.end());
  return work;
}

// Rollout i's opponent leave from `belief`, at a quantile of the rollout's
// seed, for every rollout index of the budget.
std::vector<Rack> belief_leaves(const belief::LeaveSampler& belief, uint64_t seed, int rollouts) {
  std::vector<Rack> out;
  out.reserve(size_t(rollouts));
  for (int i = 0; i < rollouts; ++i) {
    const uint64_t s = (seed + uint64_t(i)) ^ kLeaveQuantileSalt;
    const double u = double(util::splitmix64(s) >> 11) * 0x1.0p-53;
    out.push_back(belief.leave(belief.sample(u)));
  }
  return out;
}

// The labeler's job for a selected position: its record but the labels, its
// group (pool, seed, per-rollout opponent leaves) and its candidates applied.
PoolJob make_job(const SimmedPosition& r, const belief::SlogLeaveBeliefs* beliefs,
                 const Dictionary& dict, int rollouts) {
  const SimPosition& p = r.position;
  PoolJob job;
  job.group = make_rollout_group(p.board, p.rack, r.base_seed);
  const std::shared_ptr<const belief::LeaveSampler> leave =
    beliefs ? beliefs->belief(r.pos) : nullptr;
  if (leave) job.group.opp_leaves = belief_leaves(*leave, r.base_seed, rollouts);
  job.record.at = r.pos;
  job.record.num_legal_moves = r.candidates.num_legal_moves;
  job.record.moves = r.candidates.moves;
  job.record.equity_ranks = r.candidates.equity_ranks;
  job.record.opp_rack = belief::summarize_opp_rack(job.group.pool, leave.get());
  Board board = p.board;
  board.ensure_movegen_caches(dict);
  for (const Move& m : r.candidates.moves) {
    job.starts.push_back(
      make_rollout_start(board, p.rack, p.scores[p.mover], p.scores[1 - p.mover], m, /*group=*/0));
  }
  return job;
}

// Prepares a file's jobs on its own thread, kPrepBatch positions at a time
// (candidate selection and job building each spread over the threads), and
// hands them out in work order. A producer failure rethrows from next(); a
// labeler failure stops the producer on destruction.
class JobProducer {
 public:
  JobProducer(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
              const belief::SlogLeaveBeliefs* beliefs, std::vector<binlog::GamePositionIndex> work);
  ~JobProducer();

  std::optional<PoolJob> next();

 private:
  void produce();
  std::vector<PoolJob> prepare(const std::vector<binlog::GamePositionIndex>& batch);

  const binlog::PendingSlog& slog_;
  const Dictionary& dict_;
  const Options& opt_;
  const belief::SlogLeaveBeliefs* beliefs_;
  std::vector<binlog::GamePositionIndex> work_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<PoolJob> ready_;
  bool finished_ = false;
  bool stopping_ = false;  // the labeler is gone; nothing will pop ready_
  std::exception_ptr error_;
  std::thread thread_;
};

JobProducer::JobProducer(const binlog::PendingSlog& slog, const Dictionary& dict,
                         const Options& opt, const belief::SlogLeaveBeliefs* beliefs,
                         std::vector<binlog::GamePositionIndex> work)
    : slog_(slog),
      dict_(dict),
      opt_(opt),
      beliefs_(beliefs),
      work_(std::move(work)),
      thread_(&JobProducer::produce, this) {}

JobProducer::~JobProducer() {
  {
    const std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

std::vector<PoolJob> JobProducer::prepare(const std::vector<binlog::GamePositionIndex>& batch) {
  SlogSimConfig config;
  config.selector = pool_selector(opt_.blank_prune_points);
  config.seed = opt_.seed;
  config.threads = opt_.threads;
  util::ProgressMeter quiet(0, "");
  const std::vector<SimmedPosition> selected =
    select_slog_candidates(slog_.bytes, dict_, config, batch, &quiet);
  std::vector<PoolJob> jobs(selected.size());
  std::vector<std::thread> workers;
  for (int t = 0; t < opt_.threads; ++t) {
    workers.emplace_back([&, t] {
      for (size_t i = size_t(t); i < selected.size(); i += size_t(opt_.threads))
        jobs[i] = make_job(selected[i], beliefs_, dict_, opt_.labeler.rollouts);
    });
  }
  for (std::thread& w : workers) w.join();
  return jobs;
}

void JobProducer::produce() {
  try {
    for (size_t begin = 0; begin < work_.size(); begin += kPrepBatch) {
      const size_t end = std::min(work_.size(), begin + kPrepBatch);
      std::vector<PoolJob> jobs = prepare({work_.begin() + begin, work_.begin() + end});
      std::unique_lock lock(mutex_);
      // Stay about one flight ahead of the labeler: positions hold their
      // candidates' boards.
      cv_.wait(lock,
               [&] { return stopping_ || ready_.size() < size_t(opt_.labeler.max_in_flight); });
      if (stopping_) break;
      for (PoolJob& j : jobs) ready_.push_back(std::move(j));
      cv_.notify_all();
    }
  } catch (...) {
    const std::lock_guard lock(mutex_);
    error_ = std::current_exception();
  }
  const std::lock_guard lock(mutex_);
  finished_ = true;
  cv_.notify_all();
}

std::optional<PoolJob> JobProducer::next() {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [&] { return !ready_.empty() || finished_; });
  if (error_) std::rethrow_exception(error_);
  if (ready_.empty()) return std::nullopt;
  PoolJob job = std::move(ready_.front());
  ready_.pop_front();
  cv_.notify_all();
  return job;
}

// Label every sampled position of `slog` and write its .pool at the end. A
// stopped run writes nothing and redoes the file.
void pool_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
               RolloutValues& values, const PoolSources& sources, LeaveModel* leave_model) {
  std::optional<belief::SlogLeaveBeliefs> beliefs;
  if (leave_model) {
    beliefs.emplace(slog.bytes, slog.path.string(), dict, leave_model->table,
                    leave_model->served.predict_fn());
  }
  std::vector<binlog::GamePositionIndex> work = sample_work(slog.bytes, opt);
  util::ProgressMeter meter(work.size(), "positions");
  PoolWriter writer(slog.sidecar(kPoolExt).string(), sources);
  PoolLabeler labeler(values, opt.labeler);
  const auto t0 = std::chrono::steady_clock::now();
  {
    JobProducer producer(slog, dict, opt, beliefs ? &*beliefs : nullptr, std::move(work));
    labeler.run([&] { return producer.next(); },
                [&](PoolPosition&& p) {
                  writer.add_position(p);
                  meter.add_done();
                });
  }
  writer.close();
  meter.finish("pools");
  const double seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const PoolLabeler::Stats& st = labeler.stats();
  std::cerr << std::format(
    "  {}: {} launches, {} rollouts ({:.0f}% of the unraced {}), {:.1f} s, {:.0f} rollouts/s\n",
    slog.path.stem().string(), st.launches, st.tasks,
    st.full_tasks ? 100.0 * double(st.tasks) / double(st.full_tasks) : 0.0, st.full_tasks, seconds,
    double(st.tasks) / seconds);
  if (const auto* gpu = dynamic_cast<const GpuRolloutValues*>(&values)) {
    const GpuRolloutValues::Timings& t = gpu->total_timings();
    std::cerr << std::format("  GPU: rollouts {:.1f} s (the kernel {:.1f} s), leaves {:.1f} s\n",
                             t.rollout_seconds, t.kernel_seconds, t.leaf_seconds);
  }
}

std::unique_ptr<RolloutValues> make_values(const Options& opt, const Dictionary& dict) {
  if (!opt.gpu()) return std::make_unique<CpuRolloutValues>(dict, opt.threads, synthetic_leaf());
  return std::make_unique<GpuRolloutValues>(
    dict, HastyEquity::instance(),
    GpuRolloutValues::Params{.leaf_onnx = opt.leaf_model,
                             .precision = nn::parse_precision(opt.leaf_precision),
                             .leaf_rows = opt.leaf_rows});
}

void add_options(boost::program_options::options_description& desc, Options& opt) {
  namespace po = boost::program_options;
  PoolLabeler::Params& l = opt.labeler;
  desc.add_options()("help,h", "show this help and exit")(
    "slog-dir", po::value<std::string>(&opt.slog_dir),
    "directory of .slog files; each without a .pool gets one")(
    "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
    "explicit .slog file to process (repeatable; overrides --slog-dir)")(
    "leaf-model", po::value<std::string>(&opt.leaf_model),
    "the leaf network (a deep_bot_leaf_eval ONNX export); 'synthetic' with --backend cpu")(
    "leave-inference", po::value<std::string>(&opt.leave_inference_path),
    "seat opponents' leaves from this leave inference model (ONNX) instead of drawing their "
    "racks uniformly; each .slog needs its .forgone sidecar")(
    "backend", po::value<std::string>(&opt.backend)->default_value(opt.backend),
    "gpu, or cpu (the CPU reference with the synthetic leaf, for tests)")(
    "positions-per-game",
    po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
    "eligible turns sampled per game")("rollouts",
                                       po::value<int>(&l.rollouts)->default_value(l.rollouts),
                                       "rollouts per candidate, the budget")(
    "race", po::value<bool>(&l.race)->default_value(l.race),
    "stop a candidate once it sits three paired standard errors below the leader, checked "
    "after rounds ending at 10/20/40/70% of --rollouts")(
    "blank-prune-points",
    po::value<int>(&opt.blank_prune_points)->default_value(opt.blank_prune_points),
    "drop a blank move scoring this many points or more below the greedy play")(
    "launch-tasks", po::value<int>(&l.launch_tasks)->default_value(l.launch_tasks),
    "rollouts per GPU launch the labeler fills toward from the positions in flight")(
    "max-in-flight", po::value<int>(&l.max_in_flight)->default_value(l.max_in_flight),
    "positions in flight at most (each holds its candidates' boards)")(
    "leaf-precision",
    po::value<std::string>(&opt.leaf_precision)->default_value(opt.leaf_precision),
    "TensorRT precision for the leaf network: fp32, fp16 or bf16")(
    "leaf-rows", po::value<int>(&opt.leaf_rows)->default_value(opt.leaf_rows),
    "rows per leaf network call")(
    "threads", po::value<int>(&opt.threads)->default_value(opt.threads),
    "CPU threads: position preparation, and the cpu backend's rollouts")(
    "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
    "run seed (drives the turn sample and the rollout seeds)")(
    "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
    "process only the first N games of each file (0 = all)");
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("policy_set_pool_tool options");
    add_options(desc, opt);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());
    const std::vector<fs::path> inputs = binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files);
    std::unique_ptr<LeaveModel> leave_model;
    if (!opt.leave_inference_path.empty()) {
      check_forgone_sidecars(inputs);
      leave_model = std::make_unique<LeaveModel>(opt.leave_inference_path,
                                                 HastyEquity::instance().leave_values());
    }
    const PoolSources sources = pool_sources(opt, leave_model.get());
    check_finished_pools(inputs, sources);
    const std::vector<binlog::PendingSlog> pending =
      binlog::load_pending_slogs(inputs, kPoolExt, /*open_leaves=*/false,
                                 "{} was played with face-up leaves; pools are hidden-leaves only");
    std::cerr << std::format(
      "policy-set pools: {} file(s), {} backend, leaf {}, {} rollouts per "
      "candidate{}, opponent leaves {}\n",
      pending.size(), opt.backend, opt.leaf_model, opt.labeler.rollouts,
      opt.labeler.race ? " (raced)" : "", leave_model ? opt.leave_inference_path : "uniform");
    if (pending.empty()) return 0;
    const std::unique_ptr<RolloutValues> values = make_values(opt, dict);
    for (const binlog::PendingSlog& p : pending) {
      try {
        pool_file(p, dict, opt, *values, sources, leave_model.get());
      } catch (const util::CleanException& ex) {
        throw util::CleanException("{}: {}", p.path.stem().string(), ex.what());
      }
    }
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
