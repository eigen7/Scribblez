// sim_candidate_survey_tool: the measurement behind
// docs/plans/sim_labeled_candidates.md. How often does a Monte-Carlo sim prefer
// a move that a HastyBot static-equity top-K cut would never have offered it,
// and what kind of move is it when it does? py/scripts/sim_candidate_survey.py
// and the blind_spots dashboard workload drive it and read its output.
//
//   sim_candidate_survey_tool --slog-dir data/slogs --recipe all --max-positions 20
//   sim_candidate_survey_tool --slog-file a.slog --recipe setup --gcg-dir /tmp/setups
//
// A recipe picks the positions and candidates; all are simmed with HastyBot
// rollouts to game end.
//   - all (default): a random --max-positions of each file's training-eligible
//     turns, simming the top --cut of the equity ranking plus every legal play
//     that places no blank. The exhaustive search for what the cut hides.
//   - setup: the same, restricted to positions with a high-value setup play
//     (sim/setup_plays.h) outside the cut, and simming only the cut plus those
//     plays. A cheap search for one known kind of move.
//   - stratified: a per-game sample of turns and a stratified sample of
//     candidates (ranking head, contention zone, uniform tail, exchanges). What
//     the cut costs on average.
//
// Each position is simmed in two stages. The screen sims every candidate at
// --rollouts and picks the best --confirm-picks moves outside the cut. The best
// of hundreds of noisy estimates is biased upward (the winner's curse), so the
// confirming stage re-sims just those picks and the cut's moves at
// --confirm-rollouts, on fresh seeds, for an unbiased reading.
//
// Each .slog gets a same-stem .simsurvey.json; files that already have one are
// skipped. Per position it records the rack, scores and bag; per candidate its
// notation, equity, rank, leave, and a RolloutSummary (sim/rollout_summary.h)
// from each stage that simmed it; and for each confirmed move its paired win
// difference against every cut move. The summaries carry what a later
// classifier needs to say why a move outside the cut sims well: the opponent's
// replies score less (defense), the mover's next move scores more off its own
// tiles (setup), or only the shape of the margin distribution changes
// (variance). Results are appended to a partial file as they finish, so an
// interrupted run resumes where it stopped (see PartialSurvey).

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
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace scribblez;

namespace json = boost::json;

constexpr const char* kSurveyExt = ".simsurvey.json";
constexpr int kSurveyVersion = 3;

struct Options {
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string recipe = "all";
  bool open_leaves = false;
  int rollouts = 1000;          // per candidate, screening stage
  int confirm_rollouts = 5000;  // per move, confirming stage
  int confirm_picks = 5;        // outside-the-cut moves the confirming stage re-sims
  int solve_max_unseen = 14;    // confirming stage solves endgames at or below this many unseen
  bool race = true;             // stop clearly beaten candidates early in the screen
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
  c.keep_summaries = true;
  c.runner.rollouts = opt.rollouts;
  if (opt.race) {
    // Checkpoints at 10%, 20%, 40% and 70% of the screen. At three paired
    // standard errors below the leader over four looks, a candidate truly equal
    // to the leader is stopped well under once per hundred positions, and
    // anything stopped was never going to be a pick.
    for (const int pct : {10, 20, 40, 70})
      if (opt.rollouts * pct / 100 > 0) c.race_checkpoints.push_back(opt.rollouts * pct / 100);
    c.race_protected = opt.cut;
  }
  // Under the all recipe a position has hundreds of candidates, so one position
  // is a long job and across-position workers would idle behind the last few.
  // There the threads go inside the position instead. Results do not depend on
  // either thread count.
  const bool wide = opt.recipe == "all";
  c.runner.threads = wide ? opt.threads : 1;
  c.seed = opt.seed;
  c.threads = wide ? 1 : opt.threads;
  return c;
}

// The confirming stage: only the `chosen` moves, at --confirm-rollouts, on
// rollout seeds past the screen's. Each is paired against the cut's moves,
// which `chosen` lists first.
SlogSimConfig confirm_config(const Options& opt, const Dictionary& dict, ChosenMoves chosen) {
  SlogSimConfig c = screen_config(opt, dict);
  c.selector = chosen_selector(std::move(chosen));
  c.paired_references = opt.cut;
  c.race_checkpoints.clear();
  c.solve_endgames_max_unseen = opt.solve_max_unseen;
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
  if (opt.confirm_picks < 1) throw util::CleanException("--confirm-picks must be >= 1");
  if (opt.cut < 1 || opt.max_plays < 0 || opt.max_positions < 0)
    throw util::CleanException("--cut must be >= 1; --max-plays and --max-positions >= 0");
  const move_set_eval::StratumQuotas& q = opt.quotas;
  if (std::min({q.top, q.mid, q.tail, q.exchange}) < 0 || q.mid_rank_limit < 1)
    throw util::CleanException("quotas must be >= 0 and --mid-rank-limit >= 1");
}

// The turns a recipe considers: a per-game sample (stratified) or every
// eligible turn (all, setup).
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

// The positions of `slog` this run sims. The all and setup recipes then sample
// --max-positions of the qualifying turns.
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

// The three commonest bingo spots, as [name, count] pairs, commonest first.
json::array bingo_spots_json(const NextMoveStats& s) {
  std::vector<std::pair<uint32_t, uint16_t>> by_count;
  for (const auto& [spot, count] : s.bingo_spots) by_count.emplace_back(count, spot);
  std::sort(by_count.begin(), by_count.end(), [](const auto& a, const auto& b) {
    return std::pair(b.first, a.second) < std::pair(a.first, b.second);
  });
  if (by_count.size() > 3) by_count.resize(3);
  json::array out;
  for (const auto& [count, spot] : by_count)
    out.push_back(json::array{bingo_spot_name(spot), count});
  return out;
}

json::object to_json(const NextMoveStats& s) {
  return {{"score_sum", s.score_sum}, {"score_hist", to_json(s.score_hist)},
          {"bingos", s.bingos},       {"bingo_spots", bingo_spots_json(s)},
          {"non_plays", s.non_plays}, {"adjacent", s.adjacent}};
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
          {"end_swing_hist", to_json(s.end_swing_hist)},
          {"self_passed", s.self_passed},
          {"opp_passed", s.opp_passed}};
}

// The tiles `rack` keeps after `m`.
std::string leave_after(Rack rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  return rack.to_string();
}

double win_equity(const RolloutSummary& s) { return (s.wins + 0.5 * s.draws) / s.n; }

// The screen's best --confirm-picks candidates outside the cut (setup plays
// only, under the setup recipe), best first by win rate, then mean margin, then
// stored order. Candidates the race stopped early are excluded: they were
// already clearly below the leader.
std::vector<int> best_outside_cut(const SimmedPosition& r, const Options& opt) {
  std::vector<int> outside;
  for (size_t c = 0; c < r.summaries.size(); ++c) {
    const int32_t rank = r.candidates.equity_ranks[c];
    const bool beyond = rank < 0 || rank >= opt.cut;
    const bool wanted = opt.recipe != "setup" || r.candidates.highlighted[c];
    if (beyond && wanted && int(r.summaries[c].n) == opt.rollouts) outside.push_back(int(c));
  }
  const auto key = [&](int c) {
    const RolloutSummary& s = r.summaries[size_t(c)];
    return std::tuple(-win_equity(s), -s.delta_sum, c);
  };
  std::sort(outside.begin(), outside.end(), [&](int a, int b) { return key(a) < key(b); });
  if (int(outside.size()) > opt.confirm_picks) outside.resize(size_t(opt.confirm_picks));
  return outside;
}

// The indices the confirming stage re-sims: the cut's candidates in stored
// order, then the picks from outside it. Empty when either group is empty.
std::vector<int> confirm_indices(const SimmedPosition& r, const Options& opt) {
  std::vector<int> out;
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    const int32_t rank = r.candidates.equity_ranks[c];
    if (rank >= 0 && rank < opt.cut) out.push_back(int(c));
  }
  const std::vector<int> picks = best_outside_cut(r, opt);
  if (out.empty() || picks.empty()) return {};
  out.insert(out.end(), picks.begin(), picks.end());
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

struct SurveyedPosition {
  const SimmedPosition* screen;
  const SimmedPosition* confirm;  // null when nothing was confirmed
};

json::object candidate_json(const SimmedPosition& r, size_t c) {
  const Move& m = r.candidates.moves[c];
  return {{"move", move_notation(r.position.board, m)},
          {"display", spelled_move_notation(r.position.board, m)},  // played-through tiles spelled
          {"equity_rank", r.candidates.equity_ranks[c]},
          {"equity", r.candidates.equities[c]},
          {"score", m.score()},
          {"is_play", m.type() == MoveType::PLAY},
          {"tiles_played", m.type() == MoveType::PLAY ? m.num_glyphs() : 0},
          {"leave", leave_after(r.position.rack, m)},
          {"is_setup", bool(r.candidates.highlighted[c])},
          {"screen", to_json(r.summaries[c])}};
}

// The confirming sim, in confirm_indices order. Per re-simmed move: its index
// into the position's candidates, its summary, and its win value minus each cut
// move's over the same rollouts, as [sum, sum of squares] per cut move.
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
          {"unseen", r.unseen},
          {"confirm_solved_endgames", p.confirm != nullptr && p.confirm->solved_endgames},
          {"candidates", std::move(candidates)},
          {"confirm", confirm_json(p, opt)}};
}

json::object header_json(const Options& opt) {
  return {{"version", kSurveyVersion},
          {"recipe", opt.recipe},
          {"cut", opt.cut},
          {"rollouts", opt.rollouts},
          {"confirm_rollouts", opt.confirm_rollouts},
          {"confirm_picks", opt.confirm_picks},
          {"solve_max_unseen", opt.solve_max_unseen},
          {"race", opt.race},
          {"max_plays", opt.max_plays},
          {"max_positions", opt.max_positions},
          {"open_leaves", opt.open_leaves},
          {"seed", opt.seed},
          {"score_bin_width", kScoreBinWidth},
          {"delta_bin_width", kDeltaBinWidth},
          {"delta_bin_floor", kDeltaBinFloor},
          {"end_swing_bin_width", kEndSwingBinWidth},
          {"end_swing_bin_floor", kEndSwingBinFloor}};
}

// A file's survey in progress, kept in <stem>.simsurvey.partial.jsonl: the
// run's header on the first line, then one finished position per line. A rerun
// reads it back and skips what is already there. A different header means
// different options, whose positions must not mix, so the rerun refuses.
class PartialSurvey {
 public:
  PartialSurvey(const fs::path& final_path, const Options& opt);

  bool done(const binlog::GamePositionIndex& at) const { return lines_.contains(at); }
  size_t num_done() const { return lines_.size(); }
  void add(const binlog::GamePositionIndex& at, std::string position_json);
  // Write the final .simsurvey.json (positions in canonical order) atomically,
  // and remove the partial file.
  void finish();

 private:
  fs::path final_path_;
  fs::path partial_path_;
  std::string header_;
  std::map<binlog::GamePositionIndex, std::string> lines_;
};

PartialSurvey::PartialSurvey(const fs::path& final_path, const Options& opt)
    : final_path_(final_path),
      partial_path_(final_path),
      header_(json::serialize(header_json(opt))) {
  partial_path_.replace_extension(".partial.jsonl");
  std::ifstream in(partial_path_);
  std::string line;
  if (!std::getline(in, line)) {
    std::ofstream(partial_path_) << header_ << "\n";
    return;
  }
  if (line != header_)
    throw util::CleanException("{} was started with different options; delete it to start over",
                               partial_path_.string());
  // A line cut short by a kill fails to parse and is dropped with all after it.
  while (std::getline(in, line)) {
    boost::system::error_code ec;
    const json::value v = json::parse(line, ec);
    if (ec) break;
    lines_[{uint32_t(v.at("game").as_int64()), uint32_t(v.at("turn").as_int64())}] = line;
  }
}

void PartialSurvey::add(const binlog::GamePositionIndex& at, std::string position_json) {
  std::ofstream(partial_path_, std::ios::app) << position_json << "\n";
  lines_[at] = std::move(position_json);
}

void PartialSurvey::finish() {
  fs::path tmp = final_path_;
  tmp += ".tmp";
  {
    std::ofstream out(tmp);
    if (!out) throw util::CleanException("cannot write {}", tmp.string());
    std::string header = header_;
    header.pop_back();  // reopen the object for the positions array
    out << header << ",\"positions\":[";
    bool first = true;
    for (const auto& [at, line] : lines_) {
      out << (first ? "\n" : ",\n") << line;
      first = false;
    }
    out << "\n]}\n";
  }
  fs::rename(tmp, final_path_);
  fs::remove(partial_path_);
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

// Write the game through the surveyed turn's played move as
// <stem>-g<game>-turn<N>.gcg, where N is the turn's 1-based number (what
// neural_rank_tool --turn takes). Setting final_scores to the running scores
// keeps the writer from emitting end-of-game rack lines for an unfinished game.
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

// Pair each screened position with its confirming sim. Both lists are in work
// order; the confirm list omits positions that had nothing to confirm.
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

// Run both stages over a batch of positions and append the results.
void survey_batch(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
                  const std::vector<binlog::GamePositionIndex>& batch, util::ProgressMeter* meter,
                  PartialSurvey* partial) {
  util::ProgressMeter quiet(0, "");  // the confirming stage rides on the screen's tick
  const std::vector<SimmedPosition> screen =
    sim_slog_positions(slog.bytes, dict, screen_config(opt, dict), batch, meter);
  ChosenMoves chosen = chosen_moves(screen, opt);
  std::vector<binlog::GamePositionIndex> confirm_work;
  for (const auto& [at, moves] : chosen) confirm_work.push_back(at);
  const std::vector<SimmedPosition> confirm = sim_slog_positions(
    slog.bytes, dict, confirm_config(opt, dict, std::move(chosen)), confirm_work, &quiet);

  for (const SurveyedPosition& p : join_stages(screen, confirm)) {
    if (!opt.gcg_dir.empty()) write_position_gcg(slog, *p.screen, opt);
    partial->add(p.screen->pos, json::serialize(position_json(p, opt)));
  }
}

// Survey in batches, appending each batch to the partial file, so a stopped run
// loses at most one batch. The all recipe threads inside a position, so its
// batch is one position; the others thread across positions and need batches
// wide enough to keep every worker busy.
void survey_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt) {
  PartialSurvey partial(slog.sidecar(kSurveyExt), opt);
  std::vector<binlog::GamePositionIndex> work = survey_work(slog, dict, opt);
  std::erase_if(work, [&](const binlog::GamePositionIndex& at) { return partial.done(at); });
  if (partial.num_done() > 0)
    std::cerr << "  resuming: " << partial.num_done() << " positions already surveyed\n";

  util::ProgressMeter meter(work.size(), "positions");
  const size_t batch_size = opt.recipe == "all" ? 1 : size_t(4 * opt.threads);
  for (size_t begin = 0; begin < work.size(); begin += batch_size) {
    const size_t end = std::min(work.size(), begin + batch_size);
    survey_batch(slog, dict, opt, {work.begin() + begin, work.begin() + end}, &meter, &partial);
  }
  meter.finish("sim-survey");
  partial.finish();
}

// Prefix the file name, so a batch run's failure names both file and position.
void process_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt) {
  try {
    survey_file(slog, dict, opt);
  } catch (const util::CleanException& ex) {
    throw util::CleanException("{}: {}", slog.path.stem().string(), ex.what());
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
      "confirming stage: rollouts for each move inside the cut and each of the screen's "
      "--confirm-picks, on fresh seeds, for an unbiased reading of the moves the screen "
      "singled out")(
      "confirm-picks", po::value<int>(&opt.confirm_picks)->default_value(opt.confirm_picks),
      "moves from outside the cut the confirming stage re-sims, the screen's best first")(
      "solve-max-unseen",
      po::value<int>(&opt.solve_max_unseen)->default_value(opt.solve_max_unseen),
      "confirming stage: at positions with at most this many unseen tiles (bag + opponent's "
      "rack) the rollouts solve their endgames instead of playing them greedily (-1 = never)")(
      "race", po::value<bool>(&opt.race)->default_value(opt.race),
      "screening stage: stop a candidate early once it sits three paired standard errors below "
      "the leader (checked at 10/20/40/70% of --rollouts); the cut's moves always finish")(
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
