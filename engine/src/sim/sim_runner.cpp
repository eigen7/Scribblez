#include "sim/sim_runner.h"

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_agent.h"
#include "agent/hasty_bot.h"
#include "encoding/game_state_encoder.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/win_pct_table.h"
#include "sim/win_table_feature.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/work_pool.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace scribblez {

namespace {

// A candidate applied to the decision point: the state handed to
// Game::play_from for its rollouts. The opponent's rack is left empty here;
// each rollout seats it.
struct AppliedCandidate {
  Board board;
  std::array<int, 2> scores{0, 0};
  std::array<Rack, 2> known_racks;
  Rack returned_to_bag;  // an EXCHANGE's surrendered tiles; empty otherwise
};

AppliedCandidate apply_candidate(const SimPosition& pos, const Move& m) {
  AppliedCandidate a;
  a.board = pos.board;
  a.scores = pos.scores;
  Rack leave = pos.rack;
  if (m.type() == MoveType::PLAY) {
    for (int i = 0; i < m.num_glyphs(); ++i) {
      const bool ok = leave.remove(m.glyph(i).rack_tile());
      RELEASE_ASSERT(ok);
    }
    a.board.apply(m);
    a.scores[pos.mover] += m.score();
  } else if (m.type() == MoveType::EXCHANGE) {
    for (int i = 0; i < m.num_glyphs(); ++i) {
      const Tile t = m.glyph(i).rack_tile();
      const bool ok = leave.remove(t);
      RELEASE_ASSERT(ok);
      a.returned_to_bag.add(t);
    }
  }
  a.known_racks[pos.mover] = leave;
  return a;
}

void set_terminal_outcome(int delta, Rollout* r) {
  r->p_win = delta > 0 ? 1.0 : 0.0;
  r->p_draw = delta == 0 ? 1.0 : 0.0;
  r->p_loss = delta < 0 ? 1.0 : 0.0;
  r->delta = delta;
  r->delta_sq = double(delta) * delta;
}

// Stages a truncated rollout's horizon leaf in the batcher. The leaf is the
// horizon ply's post-move, pre-draw state from its mover's point of view: the
// sample kind the position-evaluation model trains on (post_move=True in
// position_eval/trainer.py). Training covers every move type, so a PLAY,
// EXCHANGE, or PASS at the horizon are all in-distribution. The encoder is
// seeded without move history, but the candidate and the >= kMinHorizonPlies
// rollout plies fill every last-move slot before anything reads them.
void stage_horizon_leaf(const SimPosition& pos, const Move& candidate, const GameLog& log,
                        const Game& game, const InputEncodingSpec& leaf_spec, LeafBatcher* batcher,
                        size_t slot) {
  GameStateEncoder enc(leaf_spec, pos.board, pos.scores, pos.mover);
  enc.apply_move(candidate);
  for (int i = 0; i < log.num_records; ++i) enc.apply_move(log.records[i].move);
  const int horizon_mover = log.records[log.num_records - 1].player;
  float* row = batcher->next_row();
  enc.encode_input(horizon_mover, game.leave(horizon_mover), game.leave(1 - horizon_mover), row);
  batcher->add(slot, horizon_mover == pos.mover);
}

// Values a truncated rollout with the win table: P(win) for the horizon's side
// to move, on its spread plus own leave value and the tiles it cannot see,
// read from the root mover's side. Nothing is batched; a lookup is cheap.
void set_win_table_outcome(const SimPosition& pos, const Game& game, const WinPctTable& table,
                           Rollout* out) {
  const GameLog log = game.log();
  const int to_move = 1 - log.records[log.num_records - 1].player;
  const int spread = game.score(to_move) - game.score(1 - to_move);
  const int unseen = game.board().unseen_count(game.rack(to_move).size());
  const float p = table.win_prob(win_table_feature(spread, game.leave(to_move), unseen));
  const bool root_to_move = to_move == pos.mover;
  out->p_win = root_to_move ? p : 1.0 - p;
  out->p_draw = 0;
  out->p_loss = 1.0 - out->p_win;
  out->delta = root_to_move ? spread : -spread;
  out->delta_sq = out->delta * out->delta;
}

bool passed(const GameLog& log, int player) {
  for (int i = 0; i < log.num_records; ++i)
    if (log.records[i].player == player && log.records[i].move.type() == MoveType::PASS)
      return true;
  return false;
}

// The opponent's known leave in the rollout seeded `seed`: the face-up leave,
// or one drawn from the leave belief at the seed's quantile.
Rack seated_opp_leave(const SimPosition& pos, uint64_t seed) {
  if (!pos.opp_leave_belief) return pos.opp_leave;
  const double u = belief::rollout_leave_quantile(seed);
  return pos.opp_leave_belief->leave(pos.opp_leave_belief->sample(u));
}

// Plays one rollout of candidate `a`. A finished game fills `out` completely;
// a truncated one stages its horizon leaf, and the batcher completes `out`
// when it flushes.
void run_rollout(const SimPosition& pos, const AppliedCandidate& a, const Move& candidate,
                 const Dictionary& dict, HastyBot& a0, HastyBot& a1, uint64_t seed,
                 const SimRunner::Params& params, const InputEncodingSpec* leaf_spec,
                 LeafBatcher* batcher, size_t slot, Rollout* out, RolloutTrace* trace) {
  const int opponent = 1 - pos.mover;
  // Built from the pre-move board and full rack, so the pool, and with it the
  // opponent's sampled tiles, is the same for every candidate (CRN). A known
  // opp_leave is seated directly and the refill draws only the rest, which is
  // the correct conditional given the mover's information.
  TileCounts pool = pos.board.unseen_tiles(pos.rack);
  std::array<Rack, 2> known_racks = a.known_racks;
  const Rack opp_leave = seated_opp_leave(pos, seed);
  if (opp_leave.size() > 0) {
    known_racks[opponent] = opp_leave;
    const bool seated = pool.remove(opp_leave.counts());
    DEBUG_ASSERT(seated, "the opponent's leave is not in the unseen pool");
  }
  Game game(a0, a1, dict, seed);
  if (params.horizon_plies > 0) game.set_max_plies(params.horizon_plies);
  game.play_from(a.board, a.scores, known_racks, pool, /*to_move=*/opponent, a.returned_to_bag);
  const GameLog log = game.log();

  out->opp_rack = log.initial_racks[opponent];
  if (trace) {
    trace->initial_racks = log.initial_racks;
    trace->turns.assign(log.records, log.records + log.num_records);
    trace->truncated = game.truncated();
  }
  if (log.num_records >= 1 && log.records[0].player == opponent)
    out->opp_reply = log.records[0].move;
  if (log.num_records >= 2 && log.records[1].player == pos.mover)
    out->self_next = log.records[1].move;
  if (!game.truncated()) {
    set_terminal_outcome(log.final_scores[pos.mover] - log.final_scores[opponent], out);
    out->self_stranded = log.final_racks[pos.mover].point_value();
    out->opp_stranded = log.final_racks[opponent].point_value();
    out->self_passed = passed(log, pos.mover);
    out->opp_passed = passed(log, opponent);
    return;
  }
  if (params.win_table) {
    set_win_table_outcome(pos, game, *params.win_table, out);
    return;
  }
  stage_horizon_leaf(pos, candidate, log, game, *leaf_spec, batcher, slot);
}

std::unique_ptr<HastyBot> make_rollout_agent(bool solve_endgames,
                                             const EndgameAgent<HastyBot>::Params& params) {
  if (solve_endgames) return std::make_unique<EndgameAgent<HastyBot>>(params);
  return std::make_unique<HastyBot>(params.base);
}

// A lane plays the slots of `results` it claims, candidate-major, so the load
// splits evenly however few rollouts there are per candidate. Lanes write
// disjoint slots, so they need no synchronization.
void run_sim_lane(const SimPosition& pos, const std::vector<AppliedCandidate>& applied,
                  const std::vector<Move>& candidates, const Dictionary& dict,
                  const SimRunner::Params& params, const InputEncodingSpec* leaf_spec,
                  util::LaneCursor& cursor, uint64_t base_seed, std::vector<Rollout>* results,
                  std::vector<RolloutTrace>* traces) {
  const int lane = int(cursor.lane());
  HastyBot::Params p0;
  p0.thread_id = lane;
  p0.name = "H0";
  HastyBot::Params p1;
  p1.thread_id = lane;
  p1.name = "H1";
  // Default temperature 0: deterministic greedy play.
  EndgameAgent<HastyBot>::Params e0, e1;
  e0.base = p0;
  e1.base = p1;
  std::unique_ptr<HastyBot> a0_owner = make_rollout_agent(params.solve_endgames, e0);
  std::unique_ptr<HastyBot> a1_owner = make_rollout_agent(params.solve_endgames, e1);
  HastyBot& a0 = *a0_owner;
  HastyBot& a1 = *a1_owner;
  std::optional<LeafBatcher> batcher;
  if (params.leaf_service) batcher.emplace(params.leaf_service, *leaf_spec, results);
  const size_t rollouts = size_t(params.rollouts);
  for (size_t slot; cursor.next(slot);) {
    const size_t c = slot / rollouts;
    const uint64_t seed = base_seed + uint64_t(slot % rollouts);
    run_rollout(pos, applied[c], candidates[c], dict, a0, a1, seed, params, leaf_spec,
                batcher ? &*batcher : nullptr, slot, &(*results)[slot],
                traces ? &(*traces)[slot] : nullptr);
  }
  if (batcher) batcher->flush();
}

}  // namespace

int end_rack_swing(const Rollout& r) {
  if (r.self_stranded == 0) return 2 * r.opp_stranded;
  if (r.opp_stranded == 0) return -2 * r.self_stranded;
  return r.opp_stranded - r.self_stranded;
}

// The opponent's reply is weighted by p_loss: the opponent wins iff the mover
// loses.
void accumulate_rollout(const Rollout& o, RolloutStats* obs) {
  ++obs->n;
  obs->wins += o.p_win;
  obs->draws += o.p_draw;
  obs->losses += o.p_loss;
  obs->delta_sum += o.delta;
  obs->delta_sq_sum += o.delta_sq;

  const int opp_cls = footprint_class(o.opp_reply);
  ++obs->opp_next_count[opp_cls];
  obs->opp_win_count[opp_cls] += float(o.p_loss);
  const int self_cls = footprint_class(o.self_next);
  ++obs->self_next_count[self_cls];
  obs->self_win_count[self_cls] += float(o.p_win);
}

void LeafBatcher::add(size_t slot, bool root_pov) {
  pending_.push_back({slot, root_pov});
  if (int(pending_.size()) == kRows) flush();
}

void LeafBatcher::flush() {
  if (pending_.empty()) return;
  const nn::PositionEvaluationSpec::Batch batch{rows_.data(), int(pending_.size())};
  const std::array<float*, 2> heads = {wld_.data(), sd_.data()};
  service_->evaluate(batch, heads);
  for (size_t j = 0; j < pending_.size(); ++j) {
    const float* wld = wld_.data() + j * nn::WldOutput::kRowElems;
    const float* sd = sd_.data() + j * nn::ScoreDiffOutput::kRowElems;
    // A non-finite readout would flow silently into training data and
    // decisions, so it is a hard error. The check is isfinite, not isnan: an
    // overflow reaches inf first, and unlike the WLD head the score-diff head
    // has no softmax to turn that inf into a NaN. It should not happen at the
    // default BF16 precision, whose exponent range matches FP32, so a trip
    // means an off-distribution input or a broken model.
    if (!std::isfinite(wld[0]) || !std::isfinite(wld[1]) || !std::isfinite(wld[2]) ||
        !std::isfinite(sd[0]) || !std::isfinite(sd[1])) {
      throw util::Exception(
        "sim runner: the leaf model returned a non-finite value at a rollout horizon "
        "(off-distribution input, or a broken model)");
    }
    Rollout& r = (*results_)[pending_[j].slot];
    if (pending_[j].root_pov) {
      r.p_win = wld[0];
      r.p_draw = wld[1];
      r.p_loss = wld[2];
      r.delta = sd[0];
    } else {
      r.p_win = wld[2];
      r.p_draw = wld[1];
      r.p_loss = wld[0];
      r.delta = -sd[0];
    }
    // The second moment is sign-invariant, so no POV branch.
    r.delta_sq = double(sd[0]) * sd[0] + double(sd[1]) * sd[1];
  }
  pending_.clear();
}

double sim_objective_value(const RolloutStats& o, SimObjective objective) {
  if (o.n == 0) return 0.0;
  const double n = o.n;
  if (objective == SimObjective::kWinRate) return (o.wins + 0.5 * o.draws) / n;
  return double(o.delta_sum) / n;
}

int best_observation_index(const std::vector<RolloutStats>& observations, SimObjective objective) {
  int best = 0;
  for (size_t i = 1; i < observations.size(); ++i) {
    if (sim_objective_value(observations[i], objective) >
        sim_objective_value(observations[best], objective)) {
      best = i;
    }
  }
  return best;
}

SimObjective parse_sim_objective(const std::string& name, const std::string& flag) {
  if (name == "winrate") return SimObjective::kWinRate;
  if (name == "spread") return SimObjective::kSpread;
  throw util::CleanException("{} must be 'winrate' or 'spread', got '{}'", flag, name);
}

std::vector<Move> equity_top_k(const MoveRequest& req, int k) {
  // k == 0 would return an empty candidate set, which callers read as
  // "nothing to choose from" rather than as a misconfiguration.
  if (k < 1) throw util::Exception("equity_top_k: k must be >= 1");
  std::vector<Move> candidates = generate_legal_plays(req);
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  candidates.insert(candidates.end(), exchanges.begin(), exchanges.end());
  if (candidates.empty()) return {Move::pass()};

  const std::vector<double> vals = HastyEquity::instance().equities(
    candidates, req.board, req.bag_size, req.opp_rack, req.my_rack);
  const int n = candidates.size();
  const int keep = std::min(k, n);
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                    [&](int a, int b) { return vals[a] > vals[b]; });
  std::vector<Move> top;
  top.reserve(size_t(keep));
  for (int j = 0; j < keep; ++j) top.push_back(candidates[size_t(idx[j])]);
  return top;
}

SimPosition sim_position_from(const MoveRequest& req) {
  SimPosition pos;
  pos.board = req.board;
  // Seat the mover as player 0, so the agent need not know its own seat.
  pos.mover = 0;
  pos.scores = {req.my_score, req.opp_score};
  pos.rack = req.my_rack;
  // Whatever the agent legitimately knows of the opponent's rack (see
  // MoveRequest): under face-up leaves, the tiles they kept.
  pos.opp_leave = req.opp_rack;
  return pos;
}

// Zero rollouts would make best_observation_index silently return the first
// candidate every time.
void SimRunner::validate(const Params& params) {
  if (params.rollouts < 1 || params.rollouts > kMaxRollouts) {
    throw util::CleanException("sim runner: rollouts must be in [1, {}]", kMaxRollouts);
  }
  if (params.leaf_service && params.win_table)
    throw util::CleanException("sim runner: a horizon takes one leaf, a model or a win table");
  validate_horizon("sim runner", params.horizon_plies,
                   params.leaf_service != nullptr || params.win_table != nullptr);
}

void SimRunner::validate_horizon(std::string_view context, int horizon_plies, bool have_leaf) {
  if ((horizon_plies > 0) != have_leaf) {
    throw util::CleanException(
      "{}: a truncation horizon and a leaf (a model or a win table) come together (horizon 0 = "
      "terminal rollouts, no leaf)",
      context);
  }
  validate_min_horizon(context, horizon_plies);
}

void SimRunner::validate_min_horizon(std::string_view context, int horizon_plies) {
  if (horizon_plies != 0 && horizon_plies < kMinHorizonPlies) {
    throw util::CleanException("{}: the horizon must be 0 (terminal rollouts) or >= {}", context,
                               kMinHorizonPlies);
  }
}

SimRunner::Params make_runner_params(SimRunner::Params sim, int horizon_plies,
                                     nn::PositionEvalService* leaf) {
  sim.horizon_plies = horizon_plies;
  sim.leaf_service = leaf;
  return sim;
}

SimRunner::SimRunner(const Dictionary& dict, const Params& params) : dict_(dict), params_(params) {
  validate(params_);
  if (params_.leaf_service) {
    leaf_spec_ = derive_input_spec(dict_, *params_.leaf_service, "sim runner");
  }
}

std::vector<Rollout> SimRunner::run_rollouts(const SimPosition& pos,
                                             const std::vector<Move>& candidates,
                                             uint64_t base_seed) const {
  return run_rollouts(pos, candidates, base_seed, params_.rollouts);
}

std::vector<Rollout> SimRunner::run_rollouts(const SimPosition& pos,
                                             const std::vector<Move>& candidates,
                                             uint64_t base_seed, int rollouts,
                                             std::vector<RolloutTrace>* traces) const {
  if (candidates.empty()) return {};
  // A non-empty bag: the pool holds the bag plus the opponent's (up to
  // RACK_SIZE) tiles, known or not.
  DEBUG_ASSERT(pos.board.unseen_tiles(pos.rack).size() > RACK_SIZE);
  if (pos.opp_leave_belief && pos.opp_leave.size() > 0)
    throw util::Exception("SimRunner: a position has a known leave or a leave belief, not both");

  std::vector<AppliedCandidate> applied;
  applied.reserve(candidates.size());
  for (const Move& m : candidates) applied.push_back(apply_candidate(pos, m));

  Params params = params_;
  params.rollouts = rollouts;
  const InputEncodingSpec* leaf_spec = params.leaf_service ? &leaf_spec_ : nullptr;
  std::vector<Rollout> results(candidates.size() * size_t(params.rollouts));
  if (traces) traces->assign(results.size(), {});
  util::parallel_lanes(results.size(), [&](util::LaneCursor& cursor) {
    run_sim_lane(pos, applied, candidates, dict_, params, leaf_spec, cursor, base_seed, &results,
                 traces);
  });
  return results;
}

std::vector<RolloutStats> SimRunner::run(const SimPosition& pos,
                                         const std::vector<Move>& candidates,
                                         uint64_t base_seed) const {
  const std::vector<Rollout> results = run_rollouts(pos, candidates, base_seed);
  // Reduce in a fixed order: with fractional contributions, floating-point
  // sums in an order that followed the thread partition would depend on the
  // thread count.
  std::vector<RolloutStats> out(candidates.size());
  for (size_t c = 0; c < out.size(); ++c)
    for (int i = 0; i < params_.rollouts; ++i)
      accumulate_rollout(results[c * size_t(params_.rollouts) + size_t(i)], &out[c]);
  return out;
}

}  // namespace scribblez
