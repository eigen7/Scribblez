// The measurement behind docs/plans/sim_labeled_candidates.md (PR 0): how often
// does a Monte-Carlo sim prefer a candidate that a HastyBot static-equity cut
// would never have offered it?
//
// For a sampled subset of each game's training-eligible turns, the tool sims a
// stratified candidate sample (the head of the equity ranking, the contention
// zone, a uniform tail, exchanges) with terminal HastyBot rollouts, and writes
// one CSV row per candidate per replica: its equity rank and its sim outcome.
// Replicas sim the same candidates on independent rollouts: the best of 64
// noisy estimates flatters itself (the winner's curse), so a pick is made on one
// replica and valued on another. Each processed
// .slog gets a same-stem .simsurvey.csv (files that already have one are
// skipped, so an interrupted run resumes by rerunning);
// py/tools/sim_candidate_survey.py drives the run and reads the rows.
//
// The rows are analysis output, not training data: nothing ships or trains on
// them, which is why they are text and carry no placement histograms.

#include "data/binary_log.h"
#include "data/slog_sampling.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "sim/slog_position_simmer.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;

constexpr const char* kSurveyExt = ".simsurvey.csv";
constexpr const char* kCsvHeader =
  "replica,game,turn,num_legal_moves,candidate,equity_rank,is_play,move_score,rollouts,wins,draws,"
  "losses,"
  "delta_sum,delta_sq_sum\n";

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  bool open_leaves = false;
  int rollouts = 300;
  int replicas = 2;
  move_set_eval::StratumQuotas quotas{.top = 9, .mid = 22, .tail = 28, .exchange = 4};
  int positions_per_game = 1;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file
};

// Replica r sims the same positions and candidates as every other replica, on
// rollout seeds disjoint from theirs.
SlogSimConfig sim_config(const Options& opt, int replica = 0) {
  SlogSimConfig c;
  c.open_leaves = opt.open_leaves;
  c.recipe.quotas = opt.quotas;
  c.runner.rollouts = opt.rollouts;
  c.runner.threads = 1;
  c.seed = opt.seed;
  c.rollout_seed_offset = uint64_t(replica) * uint64_t(opt.rollouts);
  c.threads = opt.threads;
  return c;
}

void validate(const Options& opt) {
  SimRunner::validate(sim_config(opt).runner);
  if (opt.replicas < 1) throw util::CleanException("--replicas must be >= 1");
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  const move_set_eval::StratumQuotas& q = opt.quotas;
  if (std::min({q.top, q.mid, q.tail, q.exchange}) < 0 || q.mid_rank_limit < 1)
    throw util::CleanException("quotas must be >= 0 and --mid-rank-limit >= 1");
}

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

void write_position_rows(int replica, const SimmedPosition& r, std::ostream& out) {
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    const Move& m = r.candidates.moves[c];
    const SimObservation& o = r.observations[c];
    out << std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", replica, r.pos.game_idx,
                       r.pos.turn_idx, r.candidates.num_legal_moves, c,
                       r.candidates.equity_ranks[c], int(m.type() == MoveType::PLAY), m.score(),
                       o.n, o.wins, o.draws, o.losses, o.delta_sum, o.delta_sq_sum);
  }
}

using ReplicaResults = std::vector<std::vector<SimmedPosition>>;

// Written to a temp name and renamed, so a survey file's existence means it is
// complete -- which is what lets a rerun skip it.
void write_survey(const fs::path& path, const ReplicaResults& replicas) {
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) throw util::CleanException("cannot write {}", tmp.string());
    out << kCsvHeader;
    for (size_t r = 0; r < replicas.size(); ++r)
      for (const SimmedPosition& p : replicas[r]) write_position_rows(int(r), p, out);
  }
  fs::rename(tmp, path);
}

void process_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
                  util::ProgressMeter* meter) {
  const std::vector<binlog::GamePositionIndex> work = sample_work(slog.bytes, opt);
  try {
    ReplicaResults replicas;
    for (int r = 0; r < opt.replicas; ++r)
      replicas.push_back(sim_slog_positions(slog.bytes, dict, sim_config(opt, r), work, meter));
    write_survey(slog.sidecar(kSurveyExt), replicas);
  } catch (const util::Exception& ex) {
    throw util::Exception("{}: {}", slog.path.stem().string(), ex.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    move_set_eval::StratumQuotas& q = opt.quotas;
    po::options_description desc("sim_candidate_survey_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of .slog files; each without a .simsurvey.csv gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim with the opponent's retained leave known; required for a face-up-leaves corpus")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
      "terminal Monte-Carlo rollouts per candidate, per replica")(
      "replicas", po::value<int>(&opt.replicas)->default_value(opt.replicas),
      "independent sims of each position (same candidates, disjoint rollout seeds); two let a "
      "pick made on one be valued on the other, free of the winner's curse")(
      "quota-top", po::value<int>(&q.top)->default_value(q.top),
      "candidates from the head of the equity ranking, after the played move")(
      "quota-mid", po::value<int>(&q.mid)->default_value(q.mid),
      "candidates sampled from ranks [quota-top, mid-rank-limit)")(
      "quota-tail", po::value<int>(&q.tail)->default_value(q.tail),
      "candidates sampled from ranks [mid-rank-limit, n)")(
      "quota-exchange", po::value<int>(&q.exchange)->default_value(q.exchange),
      "candidates sampled among the exchanges")(
      "mid-rank-limit", po::value<int>(&q.mid_rank_limit)->default_value(q.mid_rank_limit),
      "exclusive rank bound of the contention zone")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed (drives position sampling, stratum sampling and rollout seeds)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all)");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);

    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());

    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), kSurveyExt, opt.open_leaves,
      "{} was played with face-up leaves; pass --open-leaves to sim it");
    if (pending.empty()) return 0;
    uint64_t total_positions = 0;
    for (const binlog::PendingSlog& p : pending)
      total_positions +=
        binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
    std::cerr << "sim-survey: " << pending.size() << " file(s), " << total_positions
              << " positions x " << opt.replicas << " replicas; " << opt.rollouts << " rollouts, "
              << opt.threads << " threads\n";

    util::ProgressMeter meter(total_positions * uint64_t(opt.replicas), "position sims");
    for (const binlog::PendingSlog& p : pending) process_file(p, dict, opt, &meter);
    meter.finish("sim-survey");
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
