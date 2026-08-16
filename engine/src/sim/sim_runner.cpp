#include "sim/sim_runner.h"

#include "agent/agent.h"
#include "agent/macondo_bot.h"
#include "game/game.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <functional>
#include <numeric>
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

// What one finished rollout contributes to a SimObservation: the final score
// delta (mover POV) plus the two moves the placement maps read. A missing
// move is represented by a default Move (PASS), which places no squares.
struct RolloutOutcome {
  int delta = 0;
  Move opp_reply{};
  Move self_next{};
};

RolloutOutcome run_rollout(const SimPosition& pos, const AppliedCandidate& a,
                           const Dictionary& dict, HastyBotAgent& a0, HastyBotAgent& a1,
                           uint64_t seed) {
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
  game.play_from(a.board, a.scores, known_racks, pool, /*to_move=*/opponent, a.returned_to_bag);
  const GameLog log = game.log();

  RolloutOutcome o;
  o.delta = log.final_scores[pos.mover] - log.final_scores[opponent];
  if (log.num_records >= 1 && log.records[0].player == opponent) o.opp_reply = log.records[0].move;
  if (log.num_records >= 2 && log.records[1].player == pos.mover) o.self_next = log.records[1].move;
  return o;
}

void accumulate(const RolloutOutcome& o, SimObservation* obs) {
  ++obs->n;
  if (o.delta > 0)
    ++obs->wins;
  else if (o.delta < 0)
    ++obs->losses;
  else
    ++obs->draws;
  obs->delta_sum += o.delta;
  obs->delta_sq_sum += int64_t(o.delta) * o.delta;

  const bool opp_won = o.delta < 0;
  const bool self_won = o.delta > 0;
  visit_placed_squares(o.opp_reply, [&](int r, int c) {
    const int cell = r * BOARD_SIZE + c;
    ++obs->opp_next_count[cell];
    if (opp_won) ++obs->opp_win_count[cell];
  });
  visit_placed_squares(o.self_next, [&](int r, int c) {
    const int cell = r * BOARD_SIZE + c;
    ++obs->self_next_count[cell];
    if (self_won) ++obs->self_win_count[cell];
  });
}

// Worker t: plays rollout indices {t, t+threads, ...} of EVERY candidate (the
// per-index seed is shared across candidates -- the CRN scheme) and
// accumulates into its own per-candidate partials.
void sim_worker(const SimPosition& pos, const std::vector<AppliedCandidate>& applied,
                const Dictionary& dict, int rollouts, int threads, int t, uint64_t base_seed,
                std::vector<SimObservation>* partial) {
  HastyBotAgent::Params p0;
  p0.thread_id = t;
  p0.name = "H0";
  HastyBotAgent::Params p1;
  p1.thread_id = t;
  p1.name = "H1";
  HastyBotAgent a0(p0), a1(p1);  // temperature 0 -> deterministic greedy argmax
  for (int i = t; i < rollouts; i += threads) {
    const uint64_t seed = base_seed + uint64_t(i);
    for (size_t c = 0; c < applied.size(); ++c) {
      const RolloutOutcome o = run_rollout(pos, applied[c], dict, a0, a1, seed);
      accumulate(o, &(*partial)[c]);
    }
  }
}

void merge(const SimObservation& from, SimObservation* into) {
  into->n += from.n;
  into->wins += from.wins;
  into->draws += from.draws;
  into->losses += from.losses;
  into->delta_sum += from.delta_sum;
  into->delta_sq_sum += from.delta_sq_sum;
  for (int i = 0; i < SimObservation::kCells; ++i) {
    into->opp_next_count[i] += from.opp_next_count[i];
    into->self_next_count[i] += from.self_next_count[i];
    into->opp_win_count[i] += from.opp_win_count[i];
    into->self_win_count[i] += from.self_win_count[i];
  }
}

}  // namespace

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
      best = int(i);
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
  const int n = int(candidates.size());
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
}

SimRunner::SimRunner(const Dictionary& dict, const Params& params) : dict_(dict), params_(params) {
  validate(params_);
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

  const int threads = std::clamp(params_.threads, 1, std::max(1, params_.rollouts));
  std::vector<std::vector<SimObservation>> partials(threads,
                                                    std::vector<SimObservation>(candidates.size()));
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(sim_worker, std::cref(pos), std::cref(applied), std::cref(dict_),
                         params_.rollouts, threads, t, base_seed, &partials[t]);
  for (auto& w : workers) w.join();

  std::vector<SimObservation> out(candidates.size());
  for (const auto& partial : partials)
    for (size_t c = 0; c < out.size(); ++c) merge(partial[c], &out[c]);
  return out;
}

}  // namespace scribblez
