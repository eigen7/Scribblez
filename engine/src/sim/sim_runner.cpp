#include "sim/sim_runner.h"

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/macondo_bot.h"
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
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

namespace scribblez {

namespace {

// A candidate applied to the decision point: the state handed to
// Game::play_from for that candidate's rollouts. PLAY places its tiles and
// scores; EXCHANGE surrenders its tiles (returned to the bag after the
// refills); PASS changes nothing. In every case the opponent's rack is left
// empty, to be sampled from the pool by the rollout's refill.
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

// Encodes the horizon leaf of a truncated rollout -- the post-move pre-draw
// state of the horizon ply's mover, from their POV, the exact sample kind the
// position evaluation model trains on -- and stages the row in the batcher,
// which completes `out` when it flushes. The trainer loads post-move rows
// (position_eval/trainer.py, post_move=True) of eligible turns of EVERY move
// type: replay_to_sampled applies the sampled PLAY, EXCHANGE, or PASS before
// encoding, so all three horizon kinds are in-distribution. The seeded
// encoder's unknown last-move slots are overwritten by the candidate and the
// >= kMinHorizonPlies rollout plies before anything reads them.
void stage_horizon_leaf(const SimPosition& pos, const Move& candidate, const GameLog& log,
                        const Game& game, const InputEncodingSpec& leaf_spec, LeafBatcher* batcher,
                        size_t slot) {
  GameStateEncoder enc(leaf_spec, pos.board, pos.scores, pos.mover);
  enc.apply_move(candidate);
  for (int i = 0; i < log.num_records; ++i) enc.apply_move(log.records[i].move);
  const int horizon_mover = log.records[log.num_records - 1].player;
  float* row = batcher->next_row();
  if (leaf_spec.opp_leave_input) {
    enc.encode_input(horizon_mover, game.leave(horizon_mover), game.leave(1 - horizon_mover),
                     /*apply_flip=*/false, row);
  } else {
    enc.encode_input(horizon_mover, game.leave(horizon_mover), /*apply_flip=*/false, row);
  }
  batcher->add(slot, horizon_mover == pos.mover);
}

// Plays one rollout of candidate `a` -- to a natural end, or to horizon_plies
// when truncating -- and fills `out`'s moves plus, for a finished game, its
// exact outcome. A truncated game instead stages the horizon state's encoded
// row in the batcher, which completes `out` when it flushes.
void run_rollout(const SimPosition& pos, const AppliedCandidate& a, const Move& candidate,
                 const Dictionary& dict, HastyBotAgent& a0, HastyBotAgent& a1, uint64_t seed,
                 int horizon_plies, const InputEncodingSpec* leaf_spec, LeafBatcher* batcher,
                 size_t slot, RolloutResult* out) {
  const int opponent = 1 - pos.mover;
  // Built from the PRE-move board and full rack so the pool -- and therefore
  // the opponent's sampled tiles -- is identical across candidates (CRN). A
  // known opp_leave seeds the opponent's retained tiles; the refill then
  // draws only their hidden replenishments from the pool, which is exactly
  // the true conditional given the mover's information (fresh draws are
  // uniform from the bag by construction).
  Bag pool = unseen_pool(pos.board, pos.rack, seed);
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
    return;
  }
  stage_horizon_leaf(pos, candidate, log, game, *leaf_spec, batcher, slot);
}

// Fold one rollout into the candidate's observation. Terminal rollouts
// contribute exact integers; truncated rollouts contribute fractional values,
// which run() reduces in a fixed order (see there for why order matters).
void accumulate(const RolloutResult& o, SimObservation* obs) {
  ++obs->n;
  obs->wins += o.p_win;
  obs->draws += o.p_draw;
  obs->losses += o.p_loss;
  obs->delta_sum += o.delta;
  obs->delta_sq_sum += o.delta_sq;

  visit_placed_squares(o.opp_reply, [&](int r, int c) {
    const int cell = r * BOARD_SIZE + c;
    ++obs->opp_next_count[cell];
    obs->opp_win_count[cell] += float(o.p_loss);
  });
  visit_placed_squares(o.self_next, [&](int r, int c) {
    const int cell = r * BOARD_SIZE + c;
    ++obs->self_next_count[cell];
    obs->self_win_count[cell] += float(o.p_win);
  });
}

// Worker t: plays rollout indices {t, t+threads, ...} of EVERY candidate (the
// per-index seed is shared across candidates -- the CRN scheme) into the flat
// results array at slot = candidate * rollouts + index. Owns its rollout
// agents and, under truncation, a LeafBatcher over the shared leaf service;
// slots are disjoint across workers, so results need no synchronization.
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
  HastyBotAgent a0(p0), a1(p1);  // temperature 0 -> deterministic greedy argmax
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

// Thread entry: runs the worker and captures any exception into *err for the
// joining thread to rethrow. A rollout's leaf guard (LeafBatcher::flush)
// throws on a non-finite readout; letting that escape a std::thread would
// call std::terminate and bypass the caller's clean error reporting.
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
    // decisions -- a NaN compares false against everything, an inf poisons
    // the running sums -- so it is a hard error. Every consumed field is
    // checked with isfinite, not isnan: an FP16 overflow reaches +/-inf
    // before any NaN, and the identity-decoded score-diff head applies no
    // softmax that would fold that inf into a NaN (unlike the WLD head), so
    // inf must be caught explicitly. It should not happen: the position
    // family's FP16 builds pin the measured overflow-prone trunk region to
    // FP32 (model_specs.h), so a trip here means a new overflow site or an
    // off-distribution input.
    if (!std::isfinite(wld[0]) || !std::isfinite(wld[1]) || !std::isfinite(wld[2]) ||
        !std::isfinite(sd[0]) || !std::isfinite(sd[1])) {
      throw util::Exception(
        "sim runner: the leaf model returned a non-finite value at a rollout horizon (new FP16 "
        "overflow site, or off-distribution input)");
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
  // Rejected as a hard error rather than tolerated: k == 0 would hand back an
  // empty candidate set, which every caller reads as "nothing to choose from"
  // rather than as a misconfiguration, and k < 0 walks partial_sort's middle
  // iterator before the range's start.
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
  // The rollouts run from the mover's point of view, so seating them as player
  // 0 costs nothing and spares the agent having to know its own seat.
  pos.mover = 0;
  pos.scores = {req.my_score, req.opp_score};
  pos.rack = req.my_rack;
  // Whatever we legitimately know of the opponent's rack (see MoveRequest):
  // under face-up leaves their retained tiles, which then seed every rollout
  // instead of being drawn from the pool.
  pos.opp_leave = req.opp_rack;
  return pos;
}

Bag unseen_pool(const Board& board, const Rack& rack, uint64_t seed) {
  Bag pool(seed);
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (!g.is_empty()) pool.remove(g.is_blank() ? BLANK : g.letter());
    }
  for (int i = 0; i < rack.size(); ++i) pool.remove(rack.tiles()[i]);
  return pool;
}

// Rejected as a user error rather than tolerated: at 0 rollouts every
// observation's mean is 0/0, and those NaNs compare false against everything,
// so best_observation_index hands back the first candidate every time and the
// caller stops simulating without ever being told.
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

std::vector<SimObservation> SimRunner::run(const SimPosition& pos,
                                           const std::vector<Move>& candidates,
                                           uint64_t base_seed) const {
  if (candidates.empty()) return {};
  // The documented non-endgame requirement: a non-empty bag at the decision
  // point. The pool holds the bag plus the opponent's (up to RACK_SIZE)
  // tiles, known or not, so the bound is uniform across information
  // conditions.
  DEBUG_ASSERT(unseen_pool(pos.board, pos.rack, 0).size() > RACK_SIZE);

  std::vector<AppliedCandidate> applied;
  applied.reserve(candidates.size());
  for (const Move& m : candidates) applied.push_back(apply_candidate(pos, m));

  Params params = params_;
  params.threads = std::clamp(params_.threads, 1, std::max(1, params_.rollouts));
  const InputEncodingSpec* leaf_spec = params.horizon_plies > 0 ? &leaf_spec_ : nullptr;
  std::vector<RolloutResult> results(candidates.size() * size_t(params.rollouts));
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(params.threads);
  for (int t = 0; t < params.threads; ++t)
    workers.emplace_back(sim_worker, std::cref(pos), std::cref(applied), std::cref(candidates),
                         std::cref(dict_), params, leaf_spec, t, base_seed, &results, &errors[t]);
  for (auto& w : workers) w.join();
  // A worker's leaf guard may have thrown; surface the first such failure on
  // the calling thread instead of leaving it to terminate the process.
  for (const std::exception_ptr& e : errors)
    if (e) std::rethrow_exception(e);

  // Reduce in fixed (candidate, rollout index) order, whatever the thread
  // count -- with fractional contributions, a merge order that followed the
  // work partition would not be.
  std::vector<SimObservation> out(candidates.size());
  for (size_t c = 0; c < out.size(); ++c)
    for (int i = 0; i < params.rollouts; ++i)
      accumulate(results[c * size_t(params.rollouts) + size_t(i)], &out[c]);
  return out;
}

}  // namespace scribblez
