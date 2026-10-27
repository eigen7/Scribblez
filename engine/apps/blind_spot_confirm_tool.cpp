// blind_spot_confirm_tool: hidden-leaves blind spots from labeled pools, on the
// GPU (docs/blind_spots.md, "Hidden-leaves blind spots on the GPU").
//
//   blind_spot_confirm_tool --slog-dir chunk/ --leaf-model leaf_eval_gen730.onnx
//     --leave-inference leave_inference_gen1188.onnx --gcg-dir found/
//
// A blind spot is a position where a move from outside HastyBot's
// static-equity top --cut out-sims all of them. The screen is a .slog's .pool
// (policy_set_pool_tool), which simmed every legal move: a pool whose leader
// (the largest q) sits outside the cut is a candidate. The confirming sim
// re-sims the cut's moves and the pool's best --picks moves outside it (by q)
// at --rollouts each, unraced, on rollouts independent of the screen's, and
// under the pool's own leaf network and leave inference model, which the
// pool's header must name. A position is kept when the best confirmed outside
// move beats the best cut move by at least --min-gain points of win percentage.
// There is no standard-error bar besides: at the default 5000 rollouts a
// paired gain's standard error stays under a point, so the 5-point bar is
// several of them already. The report records it per position.
//
// Each .slog gets <stem>.blindspots.json: every confirmed position, kept or
// not, and how many pools the screen read, so rates stay computable. Each kept
// position's game goes to --gcg-dir (data/slog_position_gcg.h). Files with a
// .blindspots.json are skipped.

#include "belief/served_leave_model.h"
#include "belief/slog_leave_beliefs.h"
#include "data/gcg_writer.h"
#include "data/pool_log.h"
#include "data/sidecar_io.h"
#include "data/slog_position_gcg.h"
#include "data/slog_sampling.h"
#include "gpu/rollout_backend.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/pool_jobs.h"
#include "sim/pool_labeler.h"
#include "sim/slog_position_simmer.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/progress.h"
#include "util/work_pool.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace json = boost::json;
using namespace scribblez;

constexpr const char* kPoolExt = ".pool";
constexpr const char* kOutExt = ".blindspots.json";
constexpr int kOutVersion = 1;
// Added to every confirming rollout seed, so the confirmation never reuses the
// screen's rollouts, which chose its picks, whatever --seed either run took.
// Rollout i uses base + i, and a pool runs fewer than 2^16 rollouts.
constexpr uint64_t kConfirmSeedOffset = uint64_t(1) << 40;

using LeaveModel = belief::ServedLeaveTable;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  RolloutBackend rollouts;
  std::string leave_inference_path;
  std::string gcg_dir;
  int cut = 10;
  int picks = 5;
  double min_gain = 5.0;  // points of win percentage
  PoolLabeler::Params labeler{.rollouts = 5000, .race = false};
  int threads = util::default_thread_count();
  uint64_t seed = 0;
};

void validate(const Options& opt) {
  opt.rollouts.validate();
  if (opt.cut < 1) throw util::CleanException("--cut must be >= 1");
  if (opt.picks < 1) throw util::CleanException("--picks must be >= 1");
  if (opt.labeler.rollouts < 2) throw util::CleanException("--rollouts must be >= 2");
  if (opt.min_gain < 0) throw util::CleanException("--min-gain must be >= 0");
  if (opt.labeler.launch_tasks < 1 || opt.labeler.max_in_flight < 1)
    throw util::CleanException("--launch-tasks and --max-in-flight must be >= 1");
}

bool in_cut(int32_t rank, int cut) { return rank >= 0 && rank < cut; }

// One candidate blind spot from the screen: the moves to confirm (the cut's,
// best rank first, then the picks, best q first) and their screen labels.
struct Screened {
  std::vector<Move> moves;
  std::vector<double> screen_win_prob;
  std::vector<double> screen_q;
  int num_cut = 0;
};

// The candidates of one pool in `order`, appended to `out`.
void add_screened(const PoolReader::Position& p, const std::vector<int>& order, Screened& out) {
  for (const int c : order) {
    out.moves.push_back(p.candidates[c].move);
    out.screen_win_prob.push_back(p.candidates[c].win_prob);
    out.screen_q.push_back(p.candidates[c].q);
  }
}

// The pool's moves to confirm, or nullopt where its leader is in the cut.
std::optional<Screened> screen(const PoolReader::Position& p, const Options& opt) {
  const int n = int(p.header->num_candidates);
  std::vector<int> cut, outside;
  int leader = 0;
  for (int c = 0; c < n; ++c) {
    (in_cut(p.candidates[c].equity_rank, opt.cut) ? cut : outside).push_back(c);
    if (p.candidates[c].q > p.candidates[leader].q) leader = c;
  }
  if (in_cut(p.candidates[leader].equity_rank, opt.cut) || cut.empty()) return std::nullopt;
  std::sort(cut.begin(), cut.end(), [&](int a, int b) {
    return p.candidates[a].equity_rank < p.candidates[b].equity_rank;
  });
  std::stable_sort(outside.begin(), outside.end(),
                   [&](int a, int b) { return p.candidates[a].q > p.candidates[b].q; });
  if (int(outside.size()) > opt.picks) outside.resize(size_t(opt.picks));
  Screened out;
  out.num_cut = int(cut.size());
  add_screened(p, cut, out);
  add_screened(p, outside, out);
  return out;
}

std::map<binlog::GamePositionIndex, Screened> screen_file(const PoolReader& pools,
                                                          const Options& opt) {
  std::map<binlog::GamePositionIndex, Screened> out;
  for (int i = 0; i < pools.num_positions(); ++i) {
    const PoolReader::Position p = pools.position(i);
    if (std::optional<Screened> s = screen(p, opt))
      out.emplace(binlog::GamePositionIndex{p.header->game_index, p.header->turn_index},
                  std::move(*s));
  }
  return out;
}

// The confirmation has to sim what the screen simmed: the same leaf and the
// same opponent beliefs.
void check_pool_sources(const PoolReader& pools, const fs::path& pool, const Options& opt,
                        const LeaveModel* leave_model) {
  PoolSources want = PoolSources::of(pools.header());
  want.leaf_model = opt.rollouts.leaf_identity();
  want.leave_model = leave_model ? pool_model(leave_model->served.identity()) : PoolModel{};
  const char* field = PoolSources::of(pools.header()).first_difference(want);
  if (field != nullptr)
    throw util::CleanException("{} was screened with another {} than this run", pool.string(),
                               field);
}

// The best of candidates [begin, end) by confirmed win probability.
size_t best_in(const PoolPosition& r, size_t begin, size_t end) {
  size_t best = begin;
  for (size_t c = begin; c < end; ++c)
    if (r.labels[c].win_prob > r.labels[best].win_prob) best = c;
  return best;
}

// The confirmed reading: the best cut move, the best outside move, and the
// latter's paired win difference over the former (mean and standard error).
struct Verdict {
  size_t best_cut = 0;
  size_t best_outside = 0;
  double gain = 0;
  double se = 0;
  bool kept = false;
};

Verdict judge(const PoolPosition& r, int num_cut, const Options& opt) {
  Verdict v;
  v.best_cut = best_in(r, 0, size_t(num_cut));
  v.best_outside = best_in(r, size_t(num_cut), r.moves.size());
  const size_t n = size_t(opt.labeler.rollouts);
  const float* a = r.rollout_values.data() + v.best_outside * n;
  const float* b = r.rollout_values.data() + v.best_cut * n;
  double sum = 0, sq = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = double(a[i]) - double(b[i]);
    sum += d;
    sq += d * d;
  }
  v.gain = sum / double(n);
  v.se = std::sqrt(std::max(0.0, sq / double(n) - v.gain * v.gain) / double(n - 1));
  v.kept = 100 * v.gain >= opt.min_gain;
  return v;
}

// The tiles `rack` keeps after `m`.
std::string leave_after(Rack rack, const Move& m) {
  if (m.type() == MoveType::PLAY || m.type() == MoveType::EXCHANGE) {
    for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  }
  return rack.to_string();
}

json::object candidate_json(const SimmedPosition& s, const Screened& screened,
                            const PoolPosition& r, size_t c) {
  const Move& m = r.moves[c];
  return {{"move", move_notation(s.position.board, m)},
          {"display", spelled_move_notation(s.position.board, m)},
          {"equity_rank", r.equity_ranks[c]},
          {"score", m.type() == MoveType::PLAY ? m.score() : 0},
          {"leave", leave_after(s.position.rack, m)},
          {"in_cut", int(c) < screened.num_cut},
          {"screen_win_prob", screened.screen_win_prob[c]},
          {"screen_q", screened.screen_q[c]},
          {"win_prob", r.labels[c].win_prob}};
}

json::object position_json(const SimmedPosition& s, const Screened& screened, const PoolPosition& r,
                           const Verdict& v) {
  json::array candidates;
  for (size_t c = 0; c < r.moves.size(); ++c)
    candidates.push_back(candidate_json(s, screened, r, c));
  const SimPosition& p = s.position;
  return {{"game", s.pos.game_idx},
          {"turn", s.pos.turn_idx},  // 0-based; the .gcg name carries turn + 1
          {"mover", p.mover},
          {"rack", p.rack.to_string()},
          {"scores", json::array{p.scores[0], p.scores[1]}},
          {"bag_size", s.bag_size},
          {"unseen", s.unseen},
          {"kept", v.kept},
          {"best_cut", v.best_cut},
          {"best_outside", v.best_outside},
          {"gain", v.gain},
          {"gain_se", v.se},
          {"candidates", std::move(candidates)}};
}

json::object model_json(const PoolModel& m) {
  return {{"hash", m.hash}, {"tag", m.tag}, {"generation", m.generation}};
}

json::object header_json(const Options& opt, const PoolSources& sources, int screened_pools) {
  return {{"version", kOutVersion},
          {"cut", opt.cut},
          {"picks", opt.picks},
          {"rollouts", opt.labeler.rollouts},
          {"min_gain", opt.min_gain},
          {"seed", opt.seed},
          {"leaf_model", model_json(sources.leaf_model)},
          {"leave_model", model_json(sources.leave_model)},
          {"screen_rollouts", sources.rollouts},
          {"pools_screened", screened_pools}};
}

void write_gcg(const binlog::PendingSlog& slog, const SimmedPosition& s, const PoolPosition& r,
               const Verdict& v, const Options& opt) {
  const Board& board = s.position.board;
  std::vector<std::string> notes = {
    std::format("hidden-leaves blind spot: turn {}; hasty{} to move with {}", s.pos.turn_idx + 1,
                s.position.mover, s.position.rack.to_string()),
    std::format("{} (#{}) beats the hasty top {}'s best, {} (#{}), by {:.1f} +- {:.1f} points of "
                "win%",
                spelled_move_notation(board, r.moves[v.best_outside]),
                r.equity_ranks[v.best_outside] + 1, opt.cut,
                spelled_move_notation(board, r.moves[v.best_cut]), r.equity_ranks[v.best_cut] + 1,
                100 * v.gain, 100 * v.se)};
  const std::string name = slog_position_gcg_name(slog.path.stem().string(), s.pos);
  std::ofstream out(fs::path(opt.gcg_dir) / name);
  if (!out) throw util::CleanException("cannot write {} in {}", name, opt.gcg_dir);
  write_slog_position_gcg(slog.bytes, s.pos, std::move(notes), out);
}

// The replayed positions to confirm, each with its chosen moves.
std::vector<SimmedPosition> select_confirm(const binlog::PendingSlog& slog, const Dictionary& dict,
                                           const std::map<binlog::GamePositionIndex, Screened>& c,
                                           const Options& opt) {
  ChosenMoves chosen;
  std::vector<binlog::GamePositionIndex> work;
  for (const auto& [at, s] : c) {
    chosen.emplace(at, s.moves);
    work.push_back(at);
  }
  SlogSimConfig config;
  config.selector = chosen_selector(std::move(chosen));
  config.seed = opt.seed;
  config.rollout_seed_offset = kConfirmSeedOffset;
  util::ProgressMeter quiet(0, "");
  return select_slog_candidates(slog.bytes, dict, config, work, &quiet);
}

// Every confirmed record, in `jobs` order.
std::vector<PoolPosition> confirm(std::vector<PoolJob> jobs, RolloutValues& values,
                                  const Options& opt) {
  util::ProgressMeter meter(jobs.size(), "positions");
  std::vector<PoolPosition> out;
  size_t next = 0;
  PoolLabeler labeler(values, opt.labeler);
  labeler.run(
    [&]() -> std::optional<PoolJob> {
      if (next == jobs.size()) return std::nullopt;
      return std::move(jobs[next++]);
    },
    [&](PoolPosition&& p) {
      out.push_back(std::move(p));
      meter.add_done();
    });
  meter.finish("confirmed");
  return out;
}

// Each confirmed position's JSON, writing the kept ones' .gcg files and
// counting them in `kept`.
json::array judge_all(const binlog::PendingSlog& slog, const std::vector<SimmedPosition>& selected,
                      const std::map<binlog::GamePositionIndex, Screened>& candidates,
                      const std::vector<PoolPosition>& records, const Options& opt, int* kept) {
  json::array positions;
  for (size_t i = 0; i < selected.size(); ++i) {
    const Screened& s = candidates.at(selected[i].pos);
    const Verdict v = judge(records[i], s.num_cut, opt);
    positions.push_back(position_json(selected[i], s, records[i], v));
    if (v.kept && !opt.gcg_dir.empty()) write_gcg(slog, selected[i], records[i], v, opt);
    *kept += v.kept;
  }
  return positions;
}

// Confirms every candidate blind spot of `slog`'s pool and writes its
// .blindspots.json and kept positions' .gcg files.
void confirm_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
                  RolloutValues& values, LeaveModel* leave_model) {
  const fs::path pool_path = slog.sidecar(kPoolExt);
  const PoolReader pools(pool_path.string());
  check_pool_sources(pools, pool_path, opt, leave_model);
  const std::map<binlog::GamePositionIndex, Screened> candidates = screen_file(pools, opt);
  const std::vector<SimmedPosition> selected = select_confirm(slog, dict, candidates, opt);
  std::optional<belief::SlogLeaveBeliefs> beliefs;
  if (leave_model) {
    beliefs.emplace(slog.bytes, slog.path.string(), dict, leave_model->table,
                    leave_model->served.predict_fn());
  }
  const std::vector<PoolPosition> records =
    confirm(make_pool_jobs(selected, beliefs ? &*beliefs : nullptr, dict, opt.labeler.rollouts),
            values, opt);

  json::object out = header_json(opt, PoolSources::of(pools.header()), pools.num_positions());
  int kept = 0;
  out["positions"] = judge_all(slog, selected, candidates, records, opt, &kept);
  const std::string text = json::serialize(out);
  write_atomically(slog.sidecar(kOutExt).string(), std::vector<char>(text.begin(), text.end()));
  std::cerr << std::format("  {}: {} pools, {} confirmed, {} kept\n", slog.path.stem().string(),
                           pools.num_positions(), selected.size(), kept);
}

void check_pools_exist(const std::vector<fs::path>& slogs) {
  for (const fs::path& slog : slogs) {
    if (!fs::exists(fs::path(slog).replace_extension(kPoolExt)))
      throw util::CleanException("{} has no .pool; run policy_set_pool_tool over it first",
                                 slog.string());
  }
}

void add_options(boost::program_options::options_description& desc, Options& opt) {
  namespace po = boost::program_options;
  PoolLabeler::Params& l = opt.labeler;
  desc.add_options()("help,h", "show this help and exit")(
    "slog-dir", po::value<std::string>(&opt.slog_dir),
    "directory of .slog files with .pool sidecars; each without a .blindspots.json gets one")(
    "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
    "explicit .slog file to process (repeatable; overrides --slog-dir)")(
    "leave-inference", po::value<std::string>(&opt.leave_inference_path),
    "the leave inference model (ONNX) the pools were labeled under, if any")(
    "gcg-dir", po::value<std::string>(&opt.gcg_dir), "write each kept position's game here")(
    "cut", po::value<int>(&opt.cut)->default_value(opt.cut),
    "HastyBot's top-K moves by static equity that a move must beat")(
    "picks", po::value<int>(&opt.picks)->default_value(opt.picks),
    "moves from outside the cut, best screen q first, that the confirmation re-sims")(
    "rollouts", po::value<int>(&l.rollouts)->default_value(l.rollouts),
    "confirming rollouts per move, unraced")(
    "min-gain", po::value<double>(&opt.min_gain)->default_value(opt.min_gain),
    "keep a position whose best outside move beats the best cut move by at least this many "
    "points of win percentage")(
    "launch-tasks", po::value<int>(&l.launch_tasks)->default_value(l.launch_tasks),
    "rollouts per GPU launch the labeler fills toward from the positions in flight")(
    "max-in-flight", po::value<int>(&l.max_in_flight)->default_value(l.max_in_flight),
    "positions in flight at most")("threads",
                                   po::value<int>(&opt.threads)->default_value(opt.threads),
                                   "CPU threads, the process's budget for every parallel step")(
    "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
    "run seed (drives the confirming rollouts)");
  opt.rollouts.add_options(desc);
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("blind_spot_confirm_tool options");
    add_options(desc, opt);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);
    util::set_worker_count(opt.threads);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());
    const std::vector<fs::path> inputs = binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files);
    check_pools_exist(inputs);
    std::unique_ptr<LeaveModel> leave_model;
    if (!opt.leave_inference_path.empty()) {
      check_forgone_sidecars(inputs);
      leave_model = std::make_unique<LeaveModel>(opt.leave_inference_path,
                                                 HastyEquity::instance().leave_values());
    }
    if (!opt.gcg_dir.empty()) fs::create_directories(opt.gcg_dir);
    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      inputs, kOutExt, /*open_leaves=*/false,
      "{} was played with face-up leaves; blind spots here are hidden-leaves only");
    std::cerr << std::format(
      "blind spots: {} file(s), {} backend, leaf {}, top {} vs {} picks, {} rollouts each, "
      "opponent leaves {}\n",
      pending.size(), opt.rollouts.backend, opt.rollouts.leaf_model, opt.cut, opt.picks,
      opt.labeler.rollouts, leave_model ? opt.leave_inference_path : "uniform");
    if (pending.empty()) return 0;
    const std::unique_ptr<RolloutValues> values = opt.rollouts.make(dict);
    for (const binlog::PendingSlog& p : pending) {
      try {
        confirm_file(p, dict, opt, *values, leave_model.get());
      } catch (const util::CleanException& ex) {
        throw util::CleanException("{}: {}", p.path.stem().string(), ex.what());
      }
    }
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
