// The measurement behind docs/plans/sim_labeled_candidates.md (PR 0): how often
// does a Monte-Carlo sim prefer a candidate that a HastyBot static-equity cut
// would never have offered it?
//
// Two recipes choose the positions and candidates, both simmed with terminal
// HastyBot rollouts:
//   * stratified -- a sampled subset of each game's training-eligible turns, and
//     at each a stratified candidate sample (the head of the equity ranking, the
//     contention zone, a uniform tail, exchanges): what the cut costs on
//     average.
//   * setup -- every eligible turn is scanned for high-value setup plays
//     (sim/setup_plays.h) ranked outside the cut; a random --max-positions of
//     the turns that have one are simmed, over the top --cut plus every setup
//     play: what the cut costs where it is most likely to cost something.
//     --gcg-dir exports each simmed position's game for manual review.
//
// One CSV row is written per candidate per replica: its equity rank and its sim
// outcome. Replicas sim the same candidates on independent rollouts: the best of
// many noisy estimates flatters itself (the winner's curse), so a pick is made
// on one replica and valued on another. Each processed .slog gets a same-stem
// .simsurvey.csv (files that already have one are skipped, so an interrupted
// run resumes by rerunning); py/scripts/sim_candidate_survey.py drives the run
// and reads the rows.
//
// The rows are analysis output, not training data: nothing ships or trains on
// them, which is why they are text and carry no placement histograms.

#include "data/binary_log.h"
#include "data/gcg_writer.h"
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
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;

constexpr const char* kSurveyExt = ".simsurvey.csv";
constexpr const char* kCsvHeader =
  "replica,game,turn,num_legal_moves,candidate,equity_rank,is_play,is_setup,move,move_score,"
  "rollouts,wins,draws,losses,delta_sum,delta_sq_sum\n";

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string recipe = "stratified";
  bool open_leaves = false;
  int rollouts = 300;
  int replicas = 2;
  // The stratified recipe.
  move_set_eval::StratumQuotas quotas{.top = 9, .mid = 22, .tail = 28, .exchange = 4};
  int positions_per_game = 1;
  // The setup recipe.
  int cut = 10;             // the equity top-K always simmed
  int max_setups = 54;      // setup plays simmed beyond the cut
  int max_positions = 100;  // simmed positions per file
  std::string gcg_dir;      // export each simmed position's game
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file

  bool setup() const { return recipe == "setup"; }
};

// Replica r sims the same positions and candidates as every other replica, on
// rollout seeds disjoint from theirs.
SlogSimConfig sim_config(const Options& opt, const Dictionary& dict, int replica = 0) {
  SlogSimConfig c;
  c.open_leaves = opt.open_leaves;
  c.selector = opt.setup() ? setup_selector(dict, opt.cut, opt.max_setups)
                           : recipe_selector({.top_k = 0, .quotas = opt.quotas});
  c.runner.rollouts = opt.rollouts;
  c.runner.threads = 1;
  c.seed = opt.seed;
  c.rollout_seed_offset = uint64_t(replica) * uint64_t(opt.rollouts);
  c.threads = opt.threads;
  return c;
}

void validate(const Options& opt) {
  SimRunner::validate(SimRunner::Params{.rollouts = opt.rollouts});
  if (opt.recipe != "stratified" && opt.recipe != "setup")
    throw util::CleanException("--recipe must be stratified or setup");
  if (opt.replicas < 1) throw util::CleanException("--replicas must be >= 1");
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  if (opt.cut < 1 || opt.max_setups < 1 || opt.max_positions < 1)
    throw util::CleanException("--cut, --max-setups and --max-positions must be >= 1");
  const move_set_eval::StratumQuotas& q = opt.quotas;
  if (std::min({q.top, q.mid, q.tail, q.exchange}) < 0 || q.mid_rank_limit < 1)
    throw util::CleanException("quotas must be >= 0 and --mid-rank-limit >= 1");
}

// The turns a recipe looks at: a per-game sample, or (setup) every eligible turn.
std::vector<binlog::GamePositionIndex> sample_work(const std::vector<char>& buf,
                                                   const Options& opt) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(buf.data() + sizeof(binlog::FileHeader));
  uint32_t num_games = hdr->num_games;
  if (opt.limit_games > 0) num_games = std::min<uint32_t>(num_games, opt.limit_games);
  const int per_game = opt.setup() ? 0 : opt.positions_per_game;
  std::vector<binlog::GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g)
    binlog::sample_eligible_turns(metas[g], g, opt.seed, per_game, &work);
  std::sort(work.begin(), work.end());
  return work;
}

// The setup recipe's work list: a random max_positions of the scanned turns the
// selector accepts, in canonical order.
std::vector<binlog::GamePositionIndex> accepted_sample(const std::vector<SimmedPosition>& scanned,
                                                       const Options& opt) {
  std::vector<binlog::GamePositionIndex> accepted;
  for (const SimmedPosition& p : scanned)
    if (!p.candidates.moves.empty()) accepted.push_back(p.pos);
  std::cerr << "  " << accepted.size() << " of " << scanned.size()
            << " eligible turns have a setup play outside the cut\n";
  std::mt19937_64 rng(opt.seed);
  std::shuffle(accepted.begin(), accepted.end(), rng);
  if (int(accepted.size()) > opt.max_positions) accepted.resize(size_t(opt.max_positions));
  std::sort(accepted.begin(), accepted.end());
  return accepted;
}

// The positions of `slog` this run sims: the sampled turns, narrowed under the
// setup recipe to a sample of those the selector accepts.
std::vector<binlog::GamePositionIndex> survey_work(const binlog::PendingSlog& slog,
                                                   const Dictionary& dict, const Options& opt) {
  std::vector<binlog::GamePositionIndex> work = sample_work(slog.bytes, opt);
  if (!opt.setup()) return work;
  util::ProgressMeter meter(work.size(), "positions scanned");
  const std::vector<SimmedPosition> scanned =
    select_slog_candidates(slog.bytes, dict, sim_config(opt, dict), work, &meter);
  meter.finish("sim-survey scan");
  return accepted_sample(scanned, opt);
}

void write_position_rows(int replica, const SimmedPosition& r, std::ostream& out) {
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    const Move& m = r.candidates.moves[c];
    const SimObservation& o = r.observations[c];
    out << std::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", replica, r.pos.game_idx,
                       r.pos.turn_idx, r.candidates.num_legal_moves, c,
                       r.candidates.equity_ranks[c], int(m.type() == MoveType::PLAY),
                       int(r.candidates.highlighted[c]), move_notation(r.position.board, m),
                       m.score(), o.n, o.wins, o.draws, o.losses, o.delta_sum, o.delta_sq_sum);
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

// "K6 AC.TA (#62), ..." for the position's setup plays outside the cut.
std::string setup_plays_note(const SimmedPosition& r, int cut) {
  std::string note;
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    if (!r.candidates.highlighted[c] || r.candidates.equity_ranks[c] < cut) continue;
    note += std::format("{}{} (#{})", note.empty() ? "" : ", ",
                        move_notation(r.position.board, r.candidates.moves[c]),
                        r.candidates.equity_ranks[c] + 1);
  }
  return note;
}

// The game through the surveyed turn's played move, as
// <stem>-g<game>-turn<N>.gcg, N the surveyed turn's 1-based number, which is
// what neural_rank_tool --turn takes. Ending the log there, on its
// running scores, keeps the writer from emitting end-of-game rack adjustments
// for a game that is not over.
void write_position_gcg(const binlog::PendingSlog& slog, const SimmedPosition& r,
                        const Options& opt) {
  std::vector<TurnRecord> scratch;
  GameLog g = binlog::make_game_view(slog.bytes.data(), r.pos.game_idx, scratch, nullptr);
  g.num_records = int(r.pos.turn_idx) + 1;
  binlog::complete_turn_records(g, g.num_records, scratch);
  g.final_scores = scratch[r.pos.turn_idx].cumulative_scores;
  g.player_names = {"hasty0", "hasty1"};
  GcgWriteOptions gcg;
  gcg.lexicon_name = Lexicon::instance().name();
  gcg.notes = {
    std::format("setup survey: turn {} (neural_rank_tool --turn {}); hasty{} to move with {}",
                r.pos.turn_idx + 1, r.pos.turn_idx + 1, r.position.mover,
                r.position.rack.to_string()),
    std::format("high-value setup plays outside the hasty top {} (hasty rank): {}", opt.cut,
                setup_plays_note(r, opt.cut))};
  const std::string name =
    std::format("{}-g{}-turn{}.gcg", slog.path.stem().string(), r.pos.game_idx, r.pos.turn_idx + 1);
  std::ofstream out(fs::path(opt.gcg_dir) / name);
  if (!out) throw util::CleanException("cannot write {} in {}", name, opt.gcg_dir);
  write_game_log_gcg(g, out, gcg);
}

void survey_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt) {
  const std::vector<binlog::GamePositionIndex> work = survey_work(slog, dict, opt);
  util::ProgressMeter meter(work.size() * uint64_t(opt.replicas), "position sims");
  ReplicaResults replicas;
  for (int r = 0; r < opt.replicas; ++r)
    replicas.push_back(
      sim_slog_positions(slog.bytes, dict, sim_config(opt, dict, r), work, &meter));
  meter.finish("sim-survey");
  write_survey(slog.sidecar(kSurveyExt), replicas);
  if (opt.gcg_dir.empty()) return;
  for (const SimmedPosition& p : replicas[0]) write_position_gcg(slog, p, opt);
}

// Prepend the file, so a batch run's failure names both file and position.
void process_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt) {
  try {
    survey_file(slog, dict, opt);
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
      "recipe", po::value<std::string>(&opt.recipe)->default_value(opt.recipe),
      "stratified (sampled turns, stratified candidates) or setup (turns with a high-value "
      "setup play outside the cut; the top --cut plus every setup play)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim with the opponent's retained leave known; required for a face-up-leaves corpus")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
      "terminal Monte-Carlo rollouts per candidate, per replica")(
      "replicas", po::value<int>(&opt.replicas)->default_value(opt.replicas),
      "independent sims of each position (same candidates, disjoint rollout seeds); two let a "
      "pick made on one be valued on the other, free of the winner's curse")(
      "quota-top", po::value<int>(&q.top)->default_value(q.top),
      "stratified: candidates from the head of the equity ranking, after the played move")(
      "quota-mid", po::value<int>(&q.mid)->default_value(q.mid),
      "stratified: candidates sampled from ranks [quota-top, mid-rank-limit)")(
      "quota-tail", po::value<int>(&q.tail)->default_value(q.tail),
      "stratified: candidates sampled from ranks [mid-rank-limit, n)")(
      "quota-exchange", po::value<int>(&q.exchange)->default_value(q.exchange),
      "stratified: candidates sampled among the exchanges")(
      "mid-rank-limit", po::value<int>(&q.mid_rank_limit)->default_value(q.mid_rank_limit),
      "stratified: exclusive rank bound of the contention zone")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "stratified: eligible turns sampled per game")(
      "cut", po::value<int>(&opt.cut)->default_value(opt.cut),
      "setup: the equity top-K always simmed")(
      "max-setups", po::value<int>(&opt.max_setups)->default_value(opt.max_setups),
      "setup: setup plays simmed beyond the cut, best-ranked first")(
      "max-positions", po::value<int>(&opt.max_positions)->default_value(opt.max_positions),
      "setup: positions simmed per file, sampled from those that qualify")(
      "gcg-dir", po::value<std::string>(&opt.gcg_dir),
      "setup: write each simmed position's game here as a .gcg")(
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
    std::cerr << "sim-survey: " << pending.size() << " file(s), recipe " << opt.recipe << "; "
              << opt.replicas << " replicas x " << opt.rollouts << " rollouts, " << opt.threads
              << " threads\n";
    if (!opt.gcg_dir.empty()) fs::create_directories(opt.gcg_dir);
    for (const binlog::PendingSlog& p : pending) process_file(p, dict, opt);
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
