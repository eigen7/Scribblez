#include "agent/leaf_bot.h"

#include "agent/hasty_bot.h"
#include "sim/hasty_rollout_task.h"
#include "sim/sim_runner.h"
#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <utility>

// The production constructor and the option parsing live in
// leaf_bot_factory.cpp.

namespace scribblez {

bool empties_bag(const Move& move, int bag) {
  return move.type() == MoveType::PLAY && move.num_glyphs() >= bag;
}

std::vector<WeightedLeave> weighted_leaves(const std::vector<Rack>& draws) {
  std::vector<WeightedLeave> out;
  const double share = 1.0 / double(draws.size());
  for (const Rack& d : draws) {
    auto same =
      std::find_if(out.begin(), out.end(), [&](const WeightedLeave& w) { return w.leave == d; });
    if (same == out.end())
      out.push_back({d, share});
    else
      same->weight += share;
  }
  return out;
}

size_t best_value(const std::vector<CandidateValue>& values) {
  size_t best = 0;
  for (size_t i = 1; i < values.size(); ++i)
    if (values[i].value > values[best].value) best = i;
  return best;
}

LeafBot::LeafBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name), params_(params), backend_(std::move(backend)) {
  if (!backend_.values || !backend_.table)
    throw util::Exception("leafbot: the backend needs a leaf and a leave table");
}

void LeafBot::begin_game(const BeginGameRequest&) {
  recorder_.begin_game();
  ply_ = 0;
  ++games_;
}

void LeafBot::observe_turn(const MoveRequest& req) {
  // The rack make_move() would have recorded, which later queries' masks read.
  if (req.bag_size > 0) recorder_.own_query(req);
}

void LeafBot::observe_move(const Move& move) {
  recorder_.observe(move);
  ++ply_;
}

uint64_t LeafBot::turn_seed() const {
  return util::splitmix64(params_.seed ^ util::splitmix64((games_ << 16) | uint64_t(ply_)));
}

MoveDecision LeafBot::make_move(const MoveRequest& req) {
  // EndgameAgent solves the turns with the bag empty, so this is its fallback,
  // and every later turn is one too: no query will read the racks not
  // recorded since.
  if (req.bag_size == 0) return hasty_best_move_wmp(req);
  if (!req.opp_rack.empty())
    throw util::Exception("leafbot: hidden leaves only, but the opponent's rack is face up");
  // Every other turn records the rack, which later queries' masks read.
  const belief::LeaveQuery q = recorder_.own_query(req);
  const std::vector<Move> moves = equity_top_k(req, params_.candidates);
  if (moves.size() == 1) return moves.front();
  std::vector<Move> keeping;
  for (const Move& m : moves)
    if (!empties_bag(m, req.bag_size)) keeping.push_back(m);
  if (keeping.size() < moves.size()) {
    MoveDecision hasty(hasty_best_move_wmp(req));
    if (params_.report_values && !keeping.empty()) hasty.values = valued(req, keeping, q);
    return hasty;
  }
  MoveDecision decision(moves.front());
  decision.values = valued(req, moves, q);
  decision.move = decision.values[best_value(decision.values)].move;
  return decision;
}

std::vector<CandidateValue> LeafBot::valued(const MoveRequest& req, const std::vector<Move>& moves,
                                            const belief::LeaveQuery& q) const {
  const std::vector<double> v = values(req, moves, opp_leaves(q));
  std::vector<CandidateValue> out;
  out.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) out.push_back({moves[i], v[i]});
  return out;
}

std::vector<WeightedLeave> LeafBot::opp_leaves(const belief::LeaveQuery& q) const {
  // Before the opponent's first move, and after a move that kept nothing, the
  // leave is empty, as the leaf's training rows have it.
  if (ply_ == 0 || q.length == 0) return {{Rack{}, 1.0}};
  const std::shared_ptr<const belief::LeaveSampler> b = belief(q);
  // Stratified quantiles, jittered by the turn: each draw stands for an equal
  // share of the belief's mass, so the mean has far less spread than over
  // independent draws.
  const double jitter = double(turn_seed() >> 11) * 0x1.0p-53;
  std::vector<Rack> draws;
  draws.reserve(size_t(params_.opp_leaves));
  for (int i = 0; i < params_.opp_leaves; ++i)
    draws.push_back(b->leave(b->sample((double(i) + jitter) / double(params_.opp_leaves))));
  return weighted_leaves(draws);
}

std::shared_ptr<const belief::LeaveSampler> LeafBot::belief(const belief::LeaveQuery& q) const {
  if (backend_.predict && q.modelled())
    return belief::beliefs_of({&q, 1}, backend_.predict, *backend_.table)[0];
  // No model's belief (none loaded, or a row its training masks, such as after
  // a pass): the leaves the unseen tiles hold, weighted by their chance.
  return std::make_shared<const belief::LeaveSampler>(belief::LeaveParams{}, q.length, q.leave_pool,
                                                      *backend_.table);
}

std::vector<double> LeafBot::values(const MoveRequest& req, const std::vector<Move>& moves,
                                    const std::vector<WeightedLeave>& opp) const {
  // Each candidate's post-move board is built once (incrementally from the
  // live board's caches) and paired with every opponent leave.
  req.board.ensure_movegen_caches(req.dict);
  std::vector<CpuLeaf> leaves;
  leaves.reserve(moves.size() * opp.size());
  for (const Move& m : moves) {
    const RolloutStart s =
      make_rollout_start(req.board, req.my_rack, req.my_score, req.opp_score, m, 0);
    for (const WeightedLeave& o : opp)
      leaves.push_back({s.board, {s.leave, o.leave}, {s.my_score, s.opp_score}});
  }
  std::vector<const CpuLeaf*> ptrs;
  ptrs.reserve(leaves.size());
  for (const CpuLeaf& leaf : leaves) ptrs.push_back(&leaf);
  const std::vector<Wld> wld = backend_.values->predict_wld(ptrs);
  std::vector<double> out(moves.size(), 0.0);
  for (size_t i = 0; i < wld.size(); ++i)
    out[i / opp.size()] += opp[i % opp.size()].weight * win_value(wld[i]);
  return out;
}

}  // namespace scribblez
