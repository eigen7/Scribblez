#include "sim/sim_runner.h"

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/endgame_hasty_bot.h"
#include "agent/hasty_bot.h"
#include "encoding/game_state_encoder.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
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

void set_terminal_outcome(int delta, RolloutResult* r) {
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

bool passed(const GameLog& log, int player) {
  for (int i = 0; i < log.num_records; ++i)
    if (log.records[i].player == player && log.records[i].move.type() == MoveType::PASS)
      return true;
  return false;
}

// Plays one rollout of candidate `a`. A finished game fills `out` completely;
// a truncated one stages its horizon leaf, and the batcher completes `out`
// when it flushes.
void run_rollout(const SimPosition& pos, const AppliedCandidate& a, const Move& candidate,
                 const Dictionary& dict, HastyBotAgent& a0, HastyBotAgent& a1, uint64_t seed,
                 int horizon_plies, const InputEncodingSpec* leaf_spec, LeafBatcher* batcher,
                 size_t slot, RolloutResult* out) {
  const int opponent = 1 - pos.mover;
  // Built from the pre-move board and full rack, so the pool, and with it the
  // opponent's sampled tiles, is the same for every candidate (CRN). A known
  // opp_leave is seated directly and the refill draws only the rest, which is
  // the correct conditional given the mover's information.
  Bag pool(seed, pos.board.unseen_tiles(pos.rack));
  std::array<Rack, 2> known_racks = a.known_racks;
  if (pos.opp_leave.size() > 0) {
    known_racks[opponent] = pos.opp_leave;
    for (int i = 0; i < pos.opp_leave.size(); ++i) pool.remove(pos.opp_leave.tiles()[i]);
  }
  Game game(a0, a1, dict, seed);
  if (horizon_plies > 0) game.set_max_plies(horizon_plies);
  game.play_from(a.board, a.scores, known_racks, pool, /*to_move=*/opponent, a.returned_to_bag);
  const GameLog log = game.log();

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
  stage_horizon_leaf(pos, candidate, log, game, *leaf_spec, batcher, slot);
}

std::unique_ptr<HastyBotAgent> make_rollout_agent(bool solve_endgames,
                                                  const EndgameHastyBotAgent::Params& params) {
  if (solve_endgames) return std::make_unique<EndgameHastyBotAgent>(params);
  return std::make_unique<HastyBotAgent>(params.hasty);
}

// Worker t plays rollout indices t, t+threads, ... of every candidate. Workers
// write disjoint slots of `results`, so they need no synchronization.
void run_sim_worker(const SimPosition& pos, const std::vector<AppliedCandidate>& applied,
                    const std::vector<Move>& candidates, const Dictionary& dict,
                    SimRunner::Params params, const InputEncodingSpec* leaf_spec, int t,
                    uint64_t base_seed, std::vector<RolloutResult>* results) {
  HastyBotAgent::Params p0;
  p0.thread_id = t;
  p0.name = "H0";
  HastyBotAgent::Params p1;
  p1.thread_id = t;
  p1.name = "H1";
  // Default temperature 0: deterministic greedy play.
  EndgameHastyBotAgent::Params e0, e1;
  e0.hasty = p0;
  e1.hasty = p1;
  std::unique_ptr<HastyBotAgent> a0_owner = make_rollout_agent(params.solve_endgames, e0);
  std::unique_ptr<HastyBotAgent> a1_owner = make_rollout_agent(params.solve_endgames, e1);
  HastyBotAgent& a0 = *a0_owner;
  HastyBotAgent& a1 = *a1_owner;
  std::optional<LeafBatcher> batcher;
  if (params.horizon_plies > 0) batcher.emplace(params.leaf_service, *leaf_spec, results);
  for (int i = t; i < params.rollouts; i += params.threads) {
    const uint64_t seed = base_seed + uint64_t(i);
    for (size_t c = 0; c < applied.size(); ++c) {
      const size_t slot = c * size_t(params.rollouts) + size_t(i);
      run_rollout(pos, applied[c], candidates[c], dict, a0, a1, seed, params.horizon_plies,
                  leaf_spec, batcher ? &*batcher : nullptr, slot, &(*results)[slot]);
    }
  }
  if (batcher) batcher->flush();
}

// Thread entry: captures any exception into *err for the joining thread to
// rethrow. LeafBatcher::flush throws on a non-finite readout, and an exception
// escaping a std::thread calls std::terminate.
void sim_worker(const SimPosition& pos, const std::vector<AppliedCandidate>& applied,
                const std::vector<Move>& candidates, const Dictionary& dict,
                SimRunner::Params params, const InputEncodingSpec* leaf_spec, int t,
                uint64_t base_seed, std::vector<RolloutResult>* results, std::exception_ptr* err) {
  try {
    run_sim_worker(pos, applied, candidates, dict, params, leaf_spec, t, base_seed, results);
  } catch (...) {
    *err = std::current_exception();
  }
}

}  // namespace

int end_rack_swing(const RolloutResult& r) {
  if (r.self_stranded == 0) return 2 * r.opp_stranded;
  if (r.opp_stranded == 0) return -2 * r.self_stranded;
  return r.opp_stranded - r.self_stranded;
}

// The opponent's reply is weighted by p_loss: the opponent wins iff the mover
// loses.
void accumulate_rollout(const RolloutResult& o, SimObservation* obs) {
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
    RolloutResult& r = (*results_)[pending_[j].slot];
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

double sim_objective_value(const SimObservation& o, SimObjective objective) {
  if (o.n == 0) return 0.0;
  const double n = o.n;
  if (objective == SimObjective::kWinRate) return (o.wins + 0.5 * o.draws) / n;
  return double(o.delta_sum) / n;
}

int best_observation_index(const std::vector<SimObservation>& observations,
                           SimObjective objective) {
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
  if (params.threads < 1) throw util::CleanException("sim runner: threads must be >= 1");
  validate_horizon("sim runner", params.horizon_plies, params.leaf_service != nullptr);
}

void SimRunner::validate_horizon(std::string_view context, int horizon_plies,
                                 bool have_leaf_service) {
  if ((horizon_plies > 0) != have_leaf_service) {
    throw util::CleanException(
      "{}: a truncation horizon and a leaf model come together (horizon 0 = terminal rollouts, "
      "no model)",
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
  if (params_.horizon_plies > 0) {
    leaf_spec_ = derive_input_spec(dict_, *params_.leaf_service, "sim runner");
  }
}

std::vector<RolloutResult> SimRunner::run_rollouts(const SimPosition& pos,
                                                   const std::vector<Move>& candidates,
                                                   uint64_t base_seed) const {
  return run_rollouts(pos, candidates, base_seed, params_.rollouts);
}

std::vector<RolloutResult> SimRunner::run_rollouts(const SimPosition& pos,
                                                   const std::vector<Move>& candidates,
                                                   uint64_t base_seed, int rollouts) const {
  if (candidates.empty()) return {};
  // A non-empty bag: the pool holds the bag plus the opponent's (up to
  // RACK_SIZE) tiles, known or not.
  DEBUG_ASSERT(pos.board.unseen_tiles(pos.rack).size() > RACK_SIZE);

  std::vector<AppliedCandidate> applied;
  applied.reserve(candidates.size());
  for (const Move& m : candidates) applied.push_back(apply_candidate(pos, m));

  Params params = params_;
  params.rollouts = rollouts;
  params.threads = std::clamp(params_.threads, 1, std::max(1, rollouts));
  const InputEncodingSpec* leaf_spec = params.horizon_plies > 0 ? &leaf_spec_ : nullptr;
  std::vector<RolloutResult> results(candidates.size() * size_t(params.rollouts));
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(params.threads);
  for (int t = 0; t < params.threads; ++t)
    workers.emplace_back(sim_worker, std::cref(pos), std::cref(applied), std::cref(candidates),
                         std::cref(dict_), params, leaf_spec, t, base_seed, &results, &errors[t]);
  for (auto& w : workers) w.join();
  for (const std::exception_ptr& e : errors)
    if (e) std::rethrow_exception(e);
  return results;
}

std::vector<SimObservation> SimRunner::run(const SimPosition& pos,
                                           const std::vector<Move>& candidates,
                                           uint64_t base_seed) const {
  const std::vector<RolloutResult> results = run_rollouts(pos, candidates, base_seed);
  // Reduce in a fixed order: with fractional contributions, floating-point
  // sums in an order that followed the thread partition would depend on the
  // thread count.
  std::vector<SimObservation> out(candidates.size());
  for (size_t c = 0; c < out.size(); ++c)
    for (int i = 0; i < params_.rollouts; ++i)
      accumulate_rollout(results[c * size_t(params_.rollouts) + size_t(i)], &out[c]);
  return out;
}

}  // namespace scribblez
