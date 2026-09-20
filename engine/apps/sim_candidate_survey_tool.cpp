// The measurement behind docs/plans/sim_labeled_candidates.md (PR 0): how often
// does a Monte-Carlo sim prefer a candidate that a HastyBot static-equity cut
// would never have offered it -- and what kind of move is it when it does?
//
// Three recipes choose the positions and candidates, all simmed with terminal
// HastyBot rollouts:
//   * all (the default) -- a random --max-positions of each file's
//     training-eligible turns (0 = every one), and at each the top --cut of the
//     equity ranking plus EVERY legal play that places no blank: the exhaustive
//     search for plays the cut hides.
//   * setup -- the same, narrowed to positions with a high-value setup play
//     (sim/setup_plays.h) outside the cut, simming only the cut plus those
//     plays: the cheap search for one known kind.
//   * stratified -- a per-game sample of turns and a stratified candidate sample
//     (the head of the ranking, the contention zone, a uniform tail,
//     exchanges): what the cut costs on average.
// --gcg-dir exports each simmed position's game for manual review.
//
// Each processed .slog gets a same-stem .simsurvey.json (files that already
// have one are skipped, so an interrupted run resumes by rerunning):
// per position its rack, scores and bag; per candidate its notation, equity,
// rank and leave; and per candidate (from the screen, and again from the
// confirming sim for the moves in it) a RolloutSummary
// (sim/rollout_summary.h) -- outcome counts, the final-margin histogram, the
// end-of-game rack settlement (tiles stranded on each rack, who played out), and
// each side's next-move score histogram, bingo count and how often that move
// played off the candidate's tiles -- plus the paired win difference against
// each move inside the cut. That is what a later classifier needs to say WHY a
// move outside the cut sims well: the opponent's replies scoring less
// (defense), the mover's next move scoring more off its own tiles (setup), or
// only the margin distribution changing shape (variance).
//
// Each position is simmed in two stages. The screen sims every candidate once
// (--rollouts) and singles out its best move outside the cut; the best of
// hundreds of noisy estimates flatters itself (the winner's curse), so the
// confirming stage re-sims that one move and the cut's moves alone, longer
// (--confirm-rollouts) and on fresh seeds, for an unbiased reading.
// py/scripts/sim_candidate_survey.py drives the run and reads the files.

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

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;

namespace json = boost::json;

constexpr const char* kSurveyExt = ".simsurvey.json";
constexpr int kSurveyVersion = 2;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string recipe = "all";
  bool open_leaves = false;
  int rollouts = 1000;          // per candidate, screening stage
  int confirm_rollouts = 5000;  // per move, confirming stage
  // The stratified recipe.
  move_set_eval::StratumQuotas quotas{.top = 9, .mid = 22, .tail = 28, .exchange = 4};
  int positions_per_game = 1;
  // The all and setup recipes.
  int cut = 10;             // the equity top-K always simmed
  int max_plays = 0;        // plays simmed beyond the cut, best-ranked first; 0 = all
  int max_positions = 100;  // simmed positions per file, sampled; 0 = all
  std::string gcg_dir;      // export each simmed position's game
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;  // 0 = all games per file

  bool stratified() const { return recipe == "stratified"; }
};

// The screening stage: the recipe's candidates at --rollouts each.
SlogSimConfig screen_config(const Options& opt, const Dictionary& dict) {
  SlogSimConfig c;
  c.open_leaves = opt.open_leaves;
  if (opt.stratified()) {
    c.selector = recipe_selector({.top_k = 0, .quotas = opt.quotas});
  } else {
    const int cap = opt.max_plays > 0 ? opt.max_plays : std::numeric_limits<int>::max();
    c.selector = opt.recipe == "setup" ? setup_selector(dict, opt.cut, cap)
                                       : all_plays_selector(dict, opt.cut, opt.max_plays);
  }
  c.keep_observations = false;
  c.keep_summaries = true;
  c.runner.rollouts = opt.rollouts;
  // Hundreds of candidates a position (the all recipe) make one position a
  // long job, and across-position workers would idle behind the last few; there
  // the threads go inside the position instead. Results do not depend on either
  // thread count.
  const bool wide = opt.recipe == "all";
  c.runner.threads = wide ? opt.threads : 1;
  c.seed = opt.seed;
  c.threads = wide ? 1 : opt.threads;
  return c;
}

// The confirming stage: the `chosen` moves alone at --confirm-rollouts, on
// rollout seeds past the screen's, each paired against the cut's moves (which
// `chosen` lists first).
SlogSimConfig confirm_config(const Options& opt, const Dictionary& dict, ChosenMoves chosen) {
  SlogSimConfig c = screen_config(opt, dict);
  c.selector = chosen_selector(std::move(chosen));
  c.paired_references = opt.cut;
  c.runner.rollouts = opt.confirm_rollouts;
  c.runner.threads = opt.threads;
  c.threads = 1;
  c.rollout_seed_offset = uint64_t(opt.rollouts);
  return c;
}

void validate(const Options& opt) {
  SimRunner::validate(SimRunner::Params{.rollouts = opt.rollouts});
  if (opt.recipe != "all" && opt.recipe != "setup" && opt.recipe != "stratified")
    throw util::CleanException("--recipe must be all, setup or stratified");
  SimRunner::validate(SimRunner::Params{.rollouts = opt.confirm_rollouts});
  if (opt.positions_per_game < 1) throw util::CleanException("--positions-per-game must be >= 1");
  if (opt.cut < 1 || opt.max_plays < 0 || opt.max_positions < 0)
    throw util::CleanException("--cut must be >= 1; --max-plays and --max-positions >= 0");
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
  const int per_game = opt.stratified() ? opt.positions_per_game : 0;
  std::vector<binlog::GamePositionIndex> work;
  for (uint32_t g = 0; g < num_games; ++g)
    binlog::sample_eligible_turns(metas[g], g, opt.seed, per_game, &work);
  std::sort(work.begin(), work.end());
  return work;
}

// A random max_positions of `turns` (0 = all of them), in canonical order.
std::vector<binlog::GamePositionIndex> position_sample(std::vector<binlog::GamePositionIndex> turns,
                                                       const Options& opt) {
  std::mt19937_64 rng(opt.seed);
  std::shuffle(turns.begin(), turns.end(), rng);
  if (opt.max_positions > 0 && int(turns.size()) > opt.max_positions)
    turns.resize(size_t(opt.max_positions));
  std::sort(turns.begin(), turns.end());
  return turns;
}

// The turns of `work` the setup selector accepts, found by a selection-only pass.
std::vector<binlog::GamePositionIndex> setup_turns(
  const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
  const std::vector<binlog::GamePositionIndex>& work) {
  util::ProgressMeter meter(work.size(), "positions scanned");
  const std::vector<SimmedPosition> scanned =
    select_slog_candidates(slog.bytes, dict, screen_config(opt, dict), work, &meter);
  meter.finish("sim-survey scan");
  std::vector<binlog::GamePositionIndex> accepted;
  for (const SimmedPosition& p : scanned)
    if (!p.candidates.moves.empty()) accepted.push_back(p.pos);
  std::cerr << "  " << accepted.size() << " of " << scanned.size()
            << " eligible turns have a setup play outside the cut\n";
  return accepted;
}

// The positions of `slog` this run sims: the stratified recipe's per-game
// sample, or a sample of every eligible turn (all) or of those with a setup
// play outside the cut (setup).
std::vector<binlog::GamePositionIndex> survey_work(const binlog::PendingSlog& slog,
                                                   const Dictionary& dict, const Options& opt) {
  std::vector<binlog::GamePositionIndex> work = sample_work(slog.bytes, opt);
  if (opt.stratified()) return work;
  if (opt.recipe == "setup") work = setup_turns(slog, dict, opt, work);
  return position_sample(std::move(work), opt);
}

template <typename T, size_t N>
json::array to_json(const std::array<T, N>& values) {
  return json::array(values.begin(), values.end());
}

json::object to_json(const NextMoveStats& s) {
  return {{"score_sum", s.score_sum},
          {"score_hist", to_json(s.score_hist)},
          {"bingos", s.bingos},
          {"non_plays", s.non_plays},
          {"adjacent", s.adjacent}};
}

json::object to_json(const RolloutSummary& s) {
  return {{"n", s.n},
          {"wins", s.wins},
          {"draws", s.draws},
          {"losses", s.losses},
          {"delta_sum", s.delta_sum},
          {"delta_sq_sum", s.delta_sq_sum},
          {"delta_hist", to_json(s.delta_hist)},
          {"opp_reply", to_json(s.opp_reply)},
          {"self_next", to_json(s.self_next)},
          {"self_stranded_sum", s.self_stranded_sum},
          {"opp_stranded_sum", s.opp_stranded_sum},
          {"self_went_out", s.self_went_out},
          {"opp_went_out", s.opp_went_out},
          {"end_swing_sum", s.end_swing_sum},
          {"end_swing_hist", to_json(s.end_swing_hist)}};
}

// The tiles `rack` keeps after `m`.
std::string leave_after(Rack rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  return rack.to_string();
}

double win_equity(const RolloutSummary& s) { return (s.wins + 0.5 * s.draws) / s.n; }

// The screen's best candidate outside the cut (ties to the mean margin, then
// the better equity rank) -- under the setup recipe, its best setup play there.
// -1 when nothing lies outside the cut.
int best_outside_cut(const SimmedPosition& r, const Options& opt) {
  int best = -1;
  for (size_t c = 0; c < r.summaries.size(); ++c) {
    const int32_t rank = r.candidates.equity_ranks[c];
    const bool outside = rank < 0 || rank >= opt.cut;
    if (!outside || (opt.recipe == "setup" && !r.candidates.highlighted[c])) continue;
    const RolloutSummary& s = r.summaries[c];
    if (best < 0 ||
        std::pair(win_equity(s), s.delta_sum) >
          std::pair(win_equity(r.summaries[size_t(best)]), r.summaries[size_t(best)].delta_sum))
      best = int(c);
  }
  return best;
}

// The indices the confirming stage re-sims: the candidates inside the cut, in
// stored order, then the screen's best outside it. Empty when either is missing.
std::vector<int> confirm_indices(const SimmedPosition& r, const Options& opt) {
  std::vector<int> out;
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    const int32_t rank = r.candidates.equity_ranks[c];
    if (rank >= 0 && rank < opt.cut) out.push_back(int(c));
  }
  const int outside = best_outside_cut(r, opt);
  if (out.empty() || outside < 0) return {};
  out.push_back(outside);
  return out;
}

ChosenMoves chosen_moves(const std::vector<SimmedPosition>& screen, const Options& opt) {
  ChosenMoves chosen;
  for (const SimmedPosition& r : screen) {
    std::vector<Move>& moves = chosen[r.pos];
    for (const int c : confirm_indices(r, opt)) moves.push_back(r.candidates.moves[size_t(c)]);
    if (moves.empty()) chosen.erase(r.pos);
  }
  return chosen;
}

// A position's screen and, where it had moves on both sides of the cut, its
// confirming sim.
struct SurveyedPosition {
  const SimmedPosition* screen;
  const SimmedPosition* confirm;  // null when nothing was confirmed
};

json::object candidate_json(const SimmedPosition& r, size_t c) {
  const Move& m = r.candidates.moves[c];
  return {{"move", move_notation(r.position.board, m)},
          {"equity_rank", r.candidates.equity_ranks[c]},
          {"equity", r.candidates.equities[c]},
          {"score", m.score()},
          {"is_play", m.type() == MoveType::PLAY},
          {"tiles_played", m.type() == MoveType::PLAY ? m.num_glyphs() : 0},
          {"leave", leave_after(r.position.rack, m)},
          {"is_setup", bool(r.candidates.highlighted[c])},
          {"screen", to_json(r.summaries[c])}};
}

// The confirming sim: per re-simmed move its index into the position's
// candidates, its summary, and its win value minus each cut move's over the
// same rollout indices ([sum, sum of squares] per cut move, in listed order).
// The last entry is the screen's pick from outside the cut.
json::array confirm_json(const SurveyedPosition& p, const Options& opt) {
  json::array out;
  if (!p.confirm) return out;
  const std::vector<int> indices = confirm_indices(*p.screen, opt);
  for (size_t k = 0; k < indices.size(); ++k) {
    json::array vs_cut;
    for (const PairedWinDiff& d : p.confirm->paired[k])
      vs_cut.push_back(json::array{d.sum, d.sq_sum});
    out.push_back(json::object{{"candidate", indices[k]},
                               {"summary", to_json(p.confirm->summaries[k])},
                               {"win_diff_vs_cut", std::move(vs_cut)}});
  }
  return out;
}

json::object position_json(const SurveyedPosition& p, const Options& opt) {
  const SimmedPosition& r = *p.screen;
  json::array candidates;
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) candidates.push_back(candidate_json(r, c));
  return {{"game", r.pos.game_idx},
          {"turn", r.pos.turn_idx},  // 0-based; neural_rank_tool --turn takes turn + 1
          {"mover", r.position.mover},
          {"rack", r.position.rack.to_string()},
          {"opp_known_leave", r.position.opp_leave.to_string()},
          {"scores", json::array{r.position.scores[0], r.position.scores[1]}},
          {"bag_size", r.bag_size},
          {"num_legal_moves", r.candidates.num_legal_moves},
          {"played", move_notation(r.position.board, r.played)},
          {"candidates", std::move(candidates)},
          {"confirm", confirm_json(p, opt)}};
}

json::object header_json(const Options& opt) {
  return {{"version", kSurveyVersion},
          {"recipe", opt.recipe},
          {"cut", opt.cut},
          {"rollouts", opt.rollouts},
          {"confirm_rollouts", opt.confirm_rollouts},
          {"open_leaves", opt.open_leaves},
          {"seed", opt.seed},
          {"score_bin_width", kScoreBinWidth},
          {"delta_bin_width", kDeltaBinWidth},
          {"delta_bin_floor", kDeltaBinFloor},
          {"end_swing_bin_width", kEndSwingBinWidth},
          {"end_swing_bin_floor", kEndSwingBinFloor}};
}

// Written to a temp name and renamed, so a survey file's existence means it is
// complete -- which is what lets a rerun skip it. Positions are serialized one
// at a time: an all-plays file runs to tens of MB.
void write_survey(const fs::path& path, const Options& opt,
                  const std::vector<SurveyedPosition>& positions) {
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) throw util::CleanException("cannot write {}", tmp.string());
    std::string header = json::serialize(header_json(opt));
    header.pop_back();  // reopen the object for the positions array
    out << header << ",\"positions\":[";
    for (size_t i = 0; i < positions.size(); ++i)
      out << (i ? ",\n" : "\n") << json::serialize(position_json(positions[i], opt));
    out << "\n]}\n";
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

// Pair each screened position with its confirming sim (both in work order, the
// confirm list skipping the positions that had nothing to confirm).
std::vector<SurveyedPosition> join_stages(const std::vector<SimmedPosition>& screen,
                                          const std::vector<SimmedPosition>& confirm) {
  std::vector<SurveyedPosition> out;
  size_t k = 0;
  for (const SimmedPosition& r : screen) {
    const bool confirmed = k < confirm.size() && confirm[k].pos == r.pos;
    out.push_back({&r, confirmed ? &confirm[k++] : nullptr});
  }
  return out;
}

// Screen every candidate once, then re-sim only the cut and the screen's best
// move outside it, longer and on fresh rollouts: the screen's best-of-hundreds
// flatters itself (the winner's curse), and the confirming sim reads the one
// move it singled out without that bias.
void survey_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt) {
  const std::vector<binlog::GamePositionIndex> work = survey_work(slog, dict, opt);
  util::ProgressMeter screen_meter(work.size(), "positions screened");
  const std::vector<SimmedPosition> screen =
    sim_slog_positions(slog.bytes, dict, screen_config(opt, dict), work, &screen_meter);
  screen_meter.finish("sim-survey screen");

  ChosenMoves chosen = chosen_moves(screen, opt);
  std::vector<binlog::GamePositionIndex> confirm_work;
  for (const auto& [at, moves] : chosen) confirm_work.push_back(at);
  util::ProgressMeter confirm_meter(confirm_work.size(), "positions confirmed");
  const std::vector<SimmedPosition> confirm = sim_slog_positions(
    slog.bytes, dict, confirm_config(opt, dict, std::move(chosen)), confirm_work, &confirm_meter);
  confirm_meter.finish("sim-survey confirm");

  write_survey(slog.sidecar(kSurveyExt), opt, join_stages(screen, confirm));
  if (opt.gcg_dir.empty()) return;
  for (const SimmedPosition& p : screen) write_position_gcg(slog, p, opt);
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
      "directory of .slog files; each without a .simsurvey.json gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "recipe", po::value<std::string>(&opt.recipe)->default_value(opt.recipe),
      "all (every legal play without a blank, plus the top --cut), setup (only positions with a "
      "high-value setup play outside the cut, simming just those plays plus the cut), or "
      "stratified (a per-game turn sample and a stratified candidate sample)")(
      "open-leaves", po::bool_switch(&opt.open_leaves),
      "sim with the opponent's retained leave known; required for a face-up-leaves corpus")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
      "screening stage: terminal Monte-Carlo rollouts per candidate")(
      "confirm-rollouts",
      po::value<int>(&opt.confirm_rollouts)->default_value(opt.confirm_rollouts),
      "confirming stage: rollouts for each move inside the cut and for the screen's best move "
      "outside it, on fresh seeds -- an unbiased reading of the one move the screen singled out")(
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
      "all/setup: the equity top-K always simmed")(
      "max-plays", po::value<int>(&opt.max_plays)->default_value(opt.max_plays),
      "all/setup: plays simmed beyond the cut, best-ranked first (0 = all)")(
      "max-positions", po::value<int>(&opt.max_positions)->default_value(opt.max_positions),
      "all/setup: positions simmed per file, sampled from those that qualify (0 = all)")(
      "gcg-dir", po::value<std::string>(&opt.gcg_dir),
      "write each simmed position's game here as a .gcg")(
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
              << opt.rollouts << " screening + " << opt.confirm_rollouts << " confirming rollouts, "
              << opt.threads << " threads\n";
    if (!opt.gcg_dir.empty()) fs::create_directories(opt.gcg_dir);
    for (const binlog::PendingSlog& p : pending) process_file(p, dict, opt);
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
