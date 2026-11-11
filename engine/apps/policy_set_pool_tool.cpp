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
// --group-positions-per-game N also labels N grouped pools per game
// (sim/grouped_pools.h) into a same-stem .group (data/group_log.h): at each
// parent, Alice's --group-moves top moves and --group-racks racks sampled for
// Bob, Bob's pool at each (move, rack), raced with the file's pools, then the
// maps of each rack's leading replies over their first --group-map-sims
// rollouts in --group-map-steps steps, in launches of their own. A file is
// then done only with both sidecars.
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
#include "data/group_log.h"
#include "data/pool_log.h"
#include "data/slog_sampling.h"
#include "gpu/gpu_rollout_values.h"
#include "gpu/rollout_backend.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/grouped_pools.h"
#include "sim/pool_jobs.h"
#include "sim/pool_labeler.h"
#include "sim/rollout_values.h"
#include "sim/slog_position_simmer.h"
#include "util/exception.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/progress.h"
#include "util/work_pool.h"

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
// Positions the producer prepares per batch, ahead of the labeler.
constexpr int kPrepBatch = 16;
// Keeps the groups' parents a sample of their own.
constexpr uint64_t kGroupSampleDomain = 0x6E0B6E0B6E0B6E0BULL;

using LeaveModel = belief::ServedLeaveTable;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  RolloutBackend rollouts;
  std::string leave_inference_path;
  int positions_per_game = 2;
  PoolLabeler::Params labeler;
  int blank_prune_points = 40;
  int group_positions_per_game = 0;
  GroupParams group;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;

  bool groups() const { return group_positions_per_game > 0; }
  const char* done_ext() const { return groups() ? kGroupExtension : kPoolExt; }
};

void validate(Options& opt) {
  opt.rollouts.validate();
  if (opt.labeler.rollouts < 1) throw util::CleanException("--rollouts must be >= 1");
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  if (opt.blank_prune_points < 0) throw util::CleanException("--blank-prune-points must be >= 0");
  if (opt.labeler.launch_tasks < 1 || opt.labeler.max_in_flight < 1)
    throw util::CleanException("--launch-tasks and --max-in-flight must be >= 1");
  if (opt.group_positions_per_game < 0)
    throw util::CleanException("--group-positions-per-game must be >= 0");
  opt.group.rollouts = opt.labeler.rollouts;
  opt.group.blank_prune_points = opt.blank_prune_points;
  if (opt.groups()) opt.group.validate();
}

// Where this run's labels come from.
PoolSources pool_sources(const Options& opt, const LeaveModel* leave_model) {
  PoolSources s;
  s.leaf_model = opt.rollouts.leaf_identity();
  if (leave_model) s.leave_model = pool_model(leave_model->served.identity());
  s.rollouts = opt.labeler.rollouts;
  return s;
}

// A finished file is skipped, not redone, so it must come from this run's
// sources: pools of mixed sources in one corpus would pass for one set of
// labels. Each sidecar that exists is checked: the .pool, and with groups
// the .group, whose maps' counts must match too.
void check_finished_pools(const std::vector<fs::path>& slogs, const PoolSources& want,
                          const Options& opt) {
  for (const fs::path& slog : slogs) {
    const fs::path pool = fs::path(slog).replace_extension(kPoolExt);
    const char* field = nullptr;
    std::string which;
    if (fs::exists(pool)) {
      field = PoolSources::of(PoolReader(pool.string()).header()).first_difference(want);
      which = pool.string();
    }
    const fs::path group = fs::path(slog).replace_extension(kGroupExtension);
    if (field == nullptr && opt.groups() && fs::exists(group)) {
      const GroupReader reader(group.string());
      const GroupFileHeader& h = reader.header();
      field = PoolSources::of(h.leaf_model, h.leave_model, int(h.rollouts)).first_difference(want);
      if (field == nullptr &&
          (int(h.map_sims) != opt.group.map_sims || int(h.map_steps) != opt.group.map_steps))
        field = "maps' rollout counts";
      which = group.string();
    }
    if (field != nullptr) {
      throw util::CleanException("{} has another {} than this run; move it aside to redo it", which,
                                 field);
    }
  }
}

// The work of one file: the pool positions and the groups' parents, each
// --positions-per-game and --group-positions-per-game eligible turns of each
// game, sampled apart.
struct FileWork {
  std::vector<binlog::GamePositionIndex> pools;
  std::vector<binlog::GamePositionIndex> groups;

  size_t size() const { return pools.size() + groups.size(); }
};

FileWork sample_work(const std::vector<char>& buf, const Options& opt) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(buf.data() + sizeof(binlog::FileHeader));
  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  FileWork work;
  // A group's parent keeps more than a rack in the bag, so that Bob's pool
  // after any of Alice's moves has a bag to draw from.
  const binlog::BagRange group_bags{.min = RACK_SIZE + 1, .max = Bag::kTotalTiles};
  for (uint32_t g = 0; g < num_games; ++g) {
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &work.pools);
    if (opt.groups()) {
      binlog::sample_eligible_turns(buf.data(), g, util::splitmix64(opt.seed ^ kGroupSampleDomain),
                                    opt.group_positions_per_game, group_bags, &work.groups);
    }
  }
  std::sort(work.pools.begin(), work.pools.end());
  std::sort(work.groups.begin(), work.groups.end());
  return work;
}

// Where a labeled record goes: the .pool, or pool `index` of `group`.
struct JobTarget {
  GroupBuild* group = nullptr;
  size_t index = 0;
};

// Prepares a file's jobs on its own thread, kPrepBatch positions at a time
// (candidate selection and job building each in parallel), the pools first
// and then the groups' pools, and hands them out in work order with each
// job's target. A producer failure rethrows from next(); a labeler failure
// stops the producer on destruction. The groups live here until the file is
// written.
class JobProducer {
 public:
  JobProducer(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
              const belief::SlogLeaveBeliefs* beliefs, const BeliefModel& bob_model, FileWork work);
  ~JobProducer();

  std::optional<PoolJob> next();
  // The targets of the jobs handed out, in order; the consumer pops them.
  std::deque<JobTarget>& handed() { return handed_; }
  // The groups, in work order, once every job is out.
  const std::deque<std::unique_ptr<GroupBuild>>& groups() const { return groups_; }

 private:
  using Batch = std::vector<std::pair<JobTarget, PoolJob>>;
  using Prepare = Batch (JobProducer::*)(const std::vector<binlog::GamePositionIndex>&);

  void produce();
  Batch prepare_pools(const std::vector<binlog::GamePositionIndex>& batch);
  Batch prepare_groups(const std::vector<binlog::GamePositionIndex>& batch);
  void offer(Batch&& jobs);
  // Runs `prepare` over `work` kPrepBatch positions at a time; false once
  // stopped.
  bool produce_all(const std::vector<binlog::GamePositionIndex>& work, Prepare prepare);

  const binlog::PendingSlog& slog_;
  const Dictionary& dict_;
  const Options& opt_;
  const belief::SlogLeaveBeliefs* beliefs_;
  const BeliefModel& bob_model_;
  FileWork work_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::pair<JobTarget, PoolJob>> ready_;
  std::deque<std::unique_ptr<GroupBuild>> groups_;  // the producer's to add to
  std::deque<JobTarget> handed_;                    // the consumer's
  bool finished_ = false;
  bool stopping_ = false;  // the labeler is gone; nothing will pop ready_
  std::exception_ptr error_;
  std::thread thread_;
};

JobProducer::JobProducer(const binlog::PendingSlog& slog, const Dictionary& dict,
                         const Options& opt, const belief::SlogLeaveBeliefs* beliefs,
                         const BeliefModel& bob_model, FileWork work)
    : slog_(slog),
      dict_(dict),
      opt_(opt),
      beliefs_(beliefs),
      bob_model_(bob_model),
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

// The positions of `batch` with `selector`'s candidates.
std::vector<SimmedPosition> select(const binlog::PendingSlog& slog, const Dictionary& dict,
                                   const Options& opt, SimCandidateSelector selector,
                                   const std::vector<binlog::GamePositionIndex>& batch) {
  SlogSimConfig config;
  config.selector = std::move(selector);
  config.seed = opt.seed;
  util::ProgressMeter quiet(0, "");
  return select_slog_candidates(slog.bytes, dict, config, batch, &quiet);
}

JobProducer::Batch JobProducer::prepare_pools(const std::vector<binlog::GamePositionIndex>& batch) {
  const std::vector<SimmedPosition> selected =
    select(slog_, dict_, opt_, pool_selector(opt_.blank_prune_points), batch);
  Batch out;
  for (PoolJob& job : make_pool_jobs(selected, beliefs_, dict_, opt_.labeler.rollouts))
    out.emplace_back(JobTarget{}, std::move(job));
  return out;
}

JobProducer::Batch JobProducer::prepare_groups(
  const std::vector<binlog::GamePositionIndex>& batch) {
  const std::vector<SimmedPosition> selected =
    select(slog_, dict_, opt_, recipe_selector({.top_k = opt_.group.moves, .quotas = {}}), batch);
  Batch out;
  for (const SimmedPosition& sel : selected) {
    GroupParent parent{
      .at = sel.pos,
      .position = sel.position,
      .bag = sel.bag_size,
      .moves = sel.candidates.moves,
      .belief = beliefs_ ? beliefs_->belief(sel.pos) : nullptr,
      .earlier = beliefs_ ? beliefs_->chooser_moves(sel.pos) : std::vector<belief::ChooserMove>{}};
    auto group = std::make_unique<GroupBuild>(
      std::move(parent), bob_model_, opt_.group, dict_,
      binlog::position_seed(opt_.seed, sel.pos.game_idx, sel.pos.turn_idx));
    std::vector<PoolJob> jobs = group->pool_jobs();
    for (size_t i = 0; i < jobs.size(); ++i)
      out.emplace_back(JobTarget{group.get(), i}, std::move(jobs[i]));
    const std::lock_guard lock(mutex_);
    groups_.push_back(std::move(group));
  }
  return out;
}

void JobProducer::offer(Batch&& jobs) {
  std::unique_lock lock(mutex_);
  // Stay about one flight ahead of the labeler: positions hold their
  // candidates' boards.
  cv_.wait(lock, [&] { return stopping_ || ready_.size() < size_t(opt_.labeler.max_in_flight); });
  if (stopping_) return;
  for (auto& j : jobs) ready_.push_back(std::move(j));
  cv_.notify_all();
}

bool JobProducer::produce_all(const std::vector<binlog::GamePositionIndex>& work, Prepare prepare) {
  for (size_t begin = 0; begin < work.size(); begin += kPrepBatch) {
    const size_t end = std::min(work.size(), begin + kPrepBatch);
    offer((this->*prepare)({work.begin() + begin, work.begin() + end}));
    const std::lock_guard lock(mutex_);
    if (stopping_) return false;
  }
  return true;
}

void JobProducer::produce() {
  try {
    if (produce_all(work_.pools, &JobProducer::prepare_pools))
      produce_all(work_.groups, &JobProducer::prepare_groups);
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
  auto [target, job] = std::move(ready_.front());
  ready_.pop_front();
  handed_.push_back(target);
  cv_.notify_all();
  return std::move(job);
}

// Hands the maps' jobs of `groups` out one group at a time, and each labeled
// record back to its group.
class MapJobs {
 public:
  explicit MapJobs(const std::deque<std::unique_ptr<GroupBuild>>& groups) : groups_(groups) {}

  std::optional<PoolJob> next();
  void labeled(const PoolPosition& p);

 private:
  const std::deque<std::unique_ptr<GroupBuild>>& groups_;
  size_t group_ = 0;
  std::deque<PoolJob> pending_;
  std::deque<std::pair<GroupBuild*, size_t>> handed_;
};

std::optional<PoolJob> MapJobs::next() {
  while (pending_.empty() && group_ < groups_.size()) {
    GroupBuild* g = groups_[group_++].get();
    std::vector<PoolJob> jobs = g->map_jobs();
    for (size_t i = 0; i < jobs.size(); ++i) {
      pending_.push_back(std::move(jobs[i]));
      handed_.emplace_back(g, i);
    }
  }
  if (pending_.empty()) return std::nullopt;
  PoolJob job = std::move(pending_.front());
  pending_.pop_front();
  return job;
}

void MapJobs::labeled(const PoolPosition& p) {
  const auto [g, i] = handed_.front();
  handed_.pop_front();
  g->map_labeled(i, p);
}

// The rollouts a labeled record ran.
long long rollouts_of(const PoolPosition& p) {
  long long n = 0;
  for (const PoolLabel& l : p.labels) n += l.rollouts;
  return n;
}

// What a file's run reports: the labeler's stats, the time, and the groups'
// share of them.
struct FileReport {
  PoolLabeler::Stats pools;  // the first stage
  double seconds = 0;
  long long group_rollouts = 0;  // of the groups' pools, in the first stage
  PoolLabeler::Stats maps;       // the second stage
  double map_seconds = 0;
  size_t groups = 0;
  size_t group_bytes = 0;

  void print(const binlog::PendingSlog& slog, const Options& opt) const;
};

void FileReport::print(const binlog::PendingSlog& slog, const Options& opt) const {
  const PoolLabeler::Stats& st = pools;
  std::cerr << std::format(
    "  {}: {} launches, {} rollouts ({:.0f}% of the unraced {}), {:.1f} s, {:.0f} rollouts/s\n",
    slog.path.stem().string(), st.launches, st.tasks,
    st.full_tasks ? 100.0 * double(st.tasks) / double(st.full_tasks) : 0.0, st.full_tasks, seconds,
    double(st.tasks) / seconds);
  if (groups == 0) return;
  // The groups' pools share the first stage's launches: their share of its
  // time by rollouts, plus the maps' stage.
  const double group_seconds =
    (st.tasks ? seconds * double(group_rollouts) / double(st.tasks) : 0.0) + map_seconds;
  std::cerr << std::format(
    "  groups: {} of {} pools, {:.2f} MB and {:.2f} s each ({} rollouts each, maps {} "
    "launches, {} rollouts, {:.1f} s)\n",
    groups, opt.group.moves * opt.group.racks, double(group_bytes) / double(groups) / 1e6,
    group_seconds / double(groups), (group_rollouts + maps.tasks) / (long long)groups,
    maps.launches, maps.tasks, map_seconds);
}

// The maps' stage: every group's leading replies' maps, then the .group.
void write_groups(const JobProducer& producer, RolloutValues& values, const Options& opt,
                  const PoolSources& sources, const fs::path& path, FileReport* report) {
  const auto t0 = std::chrono::steady_clock::now();
  PoolLabeler labeler(values, {.rollouts = opt.group.map_sims,
                               .race = false,
                               .launch_tasks = opt.labeler.launch_tasks,
                               .max_in_flight = opt.labeler.max_in_flight,
                               .sum_maps = true});
  MapJobs jobs(producer.groups());
  labeler.run([&] { return jobs.next(); }, [&](PoolPosition&& p) { jobs.labeled(p); });
  report->maps = labeler.stats();
  report->map_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  GroupWriter writer(path.string(), sources, opt.group.map_sims, opt.group.map_steps);
  for (const auto& g : producer.groups()) writer.add_group(g->record());
  report->groups = producer.groups().size();
  report->group_bytes = writer.bytes();
  writer.close();
}

// Label every sampled position of `slog` and write its .pool at the end,
// then its groups and their .group. A stopped run writes nothing and redoes
// the file.
void pool_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
               RolloutValues& values, const PoolSources& sources, LeaveModel* leave_model) {
  std::optional<belief::SlogLeaveBeliefs> beliefs;
  BeliefModel bob_model;
  if (leave_model) {
    beliefs.emplace(slog.bytes, slog.path.string(), dict, leave_model->table,
                    leave_model->served.predict_fn());
    bob_model = {leave_model->served.predict_fn(), &leave_model->table};
  }
  FileWork work = sample_work(slog.bytes, opt);
  util::ProgressMeter meter(work.size(), "positions");
  PoolWriter writer(slog.sidecar(kPoolExt).string(), sources);
  PoolLabeler labeler(values, opt.labeler);
  FileReport report;
  const auto t0 = std::chrono::steady_clock::now();
  JobProducer producer(slog, dict, opt, beliefs ? &*beliefs : nullptr, bob_model, std::move(work));
  labeler.run([&] { return producer.next(); },
              [&](PoolPosition&& p) {
                const JobTarget target = producer.handed().front();
                producer.handed().pop_front();
                if (target.group == nullptr) {
                  writer.add_position(p);
                  meter.add_done();
                } else {
                  report.group_rollouts += rollouts_of(p);
                  target.group->pool_labeled(target.index, std::move(p));
                  if (target.index + 1 == size_t(opt.group.moves * opt.group.racks))
                    meter.add_done();
                }
              });
  writer.close();
  meter.finish("pools");
  report.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  report.pools = labeler.stats();
  if (opt.groups())
    write_groups(producer, values, opt, sources, slog.sidecar(kGroupExtension), &report);
  report.print(slog, opt);
  if (const auto* gpu = dynamic_cast<const GpuRolloutValues*>(&values)) {
    const GpuRolloutValues::Timings& t = gpu->total_timings();
    std::cerr << std::format(
      "  GPU: rollouts {:.1f} s (the kernel {:.1f} s), leaves {:.1f} s ({} unfinished rollouts)\n",
      t.rollout_seconds, t.kernel_seconds, t.leaf_seconds, t.leaves);
  }
}

void add_options(boost::program_options::options_description& desc, Options& opt) {
  namespace po = boost::program_options;
  PoolLabeler::Params& l = opt.labeler;
  desc.add_options()("help,h", "show this help and exit")(
    "slog-dir", po::value<std::string>(&opt.slog_dir),
    "directory of .slog files; each without a .pool gets one")(
    "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
    "explicit .slog file to process (repeatable; overrides --slog-dir)")(
    "leave-inference", po::value<std::string>(&opt.leave_inference_path),
    "seat opponents' leaves from this leave inference model (ONNX) instead of drawing their "
    "racks uniformly; each .slog needs its .forgone sidecar")(
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
    "threads", po::value<int>(&opt.threads)->default_value(opt.threads),
    "CPU threads, the process's budget for every parallel step")(
    "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
    "run seed (drives the turn sample and the rollout seeds)")(
    "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
    "process only the first N games of each file (0 = all)");
  GroupParams& g = opt.group;
  desc.add_options()(
    "group-positions-per-game",
    po::value<int>(&opt.group_positions_per_game)->default_value(opt.group_positions_per_game),
    "eligible turns per game labeled as grouped pools into a .group sidecar (0 = none)")(
    "group-moves", po::value<int>(&g.moves)->default_value(g.moves),
    "groups: the mover's static-equity top moves")(
    "group-racks", po::value<int>(&g.racks)->default_value(g.racks),
    "groups: racks sampled for the opponent, the same under every move")(
    "group-map-sims", po::value<int>(&g.map_sims)->default_value(g.map_sims),
    "groups: a search's sims per rack, the most rollouts a reply's maps sum")(
    "group-map-steps", po::value<int>(&g.map_steps)->default_value(g.map_steps),
    "groups: maps per leading reply, over its first map-sims * j / steps rollouts");
  opt.rollouts.add_options(desc);
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
    util::set_worker_count(opt.threads);

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
    check_finished_pools(inputs, sources, opt);
    const std::vector<binlog::PendingSlog> pending =
      binlog::load_pending_slogs(inputs, opt.done_ext(), /*open_leaves=*/false,
                                 "{} was played with face-up leaves; pools are hidden-leaves only");
    std::cerr << std::format(
      "policy-set pools: {} file(s), {} backend, leaf {}, {} rollouts per "
      "candidate{}, opponent leaves {}{}\n",
      pending.size(), opt.rollouts.backend, opt.rollouts.leaf_model, opt.labeler.rollouts,
      opt.labeler.race ? " (raced)" : "", leave_model ? opt.leave_inference_path : "uniform",
      opt.groups() ? std::format(", {} groups per game of {} moves x {} racks",
                                 opt.group_positions_per_game, opt.group.moves, opt.group.racks)
                   : "");
    if (pending.empty()) return 0;
    const std::unique_ptr<RolloutValues> values = opt.rollouts.make(dict);
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
