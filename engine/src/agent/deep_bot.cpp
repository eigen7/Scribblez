#include "agent/deep_bot.h"

#include "agent/hasty_bot.h"
#include "game/bag.h"
#include "sim/hasty_rollout_task.h"
#include "sim/sim_runner.h"
#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <span>
#include <utility>

// The production constructor and the option parsing live in
// deep_bot_factory.cpp.

namespace scribblez {

namespace {

// `count` racks of `length` tiles, each drawn from `pool` at random.
std::vector<Rack> random_racks(const TileCounts& pool, int length, int count, uint64_t seed) {
  Bag bag(seed, pool);
  std::vector<Rack> racks(static_cast<size_t>(count));
  for (Rack& rack : racks) {
    for (int i = 0; i < length; ++i) rack.add(bag.draw());
    for (int i = 0; i < rack.size(); ++i) bag.put_back(rack.tiles()[size_t(i)]);
  }
  return racks;
}

}  // namespace

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

std::vector<WeightedLeave> equal_weights(const std::vector<Rack>& draws) {
  std::vector<WeightedLeave> out;
  out.reserve(draws.size());
  for (const Rack& d : draws) out.push_back({d, 1.0 / double(draws.size())});
  return out;
}

size_t best_value(const std::vector<CandidateValue>& values) {
  size_t best = 0;
  for (size_t i = 1; i < values.size(); ++i)
    if (values[i].value > values[best].value) best = i;
  return best;
}

DeepBot::DeepBot(const Params& params, Backend backend)
    : Agent(params.thread_id, params.name), params_(params), backend_(std::move(backend)) {
  if (!backend_.values || !backend_.table)
    throw util::Exception("deepbot: the backend needs a leaf and a leave table");
}

void DeepBot::begin_game(const BeginGameRequest&) {
  recorder_.begin_game();
  ply_ = 0;
  ++games_;
}

void DeepBot::observe_turn(const MoveRequest& req) {
  // The rack make_move() would have recorded, which later queries' masks read.
  if (req.bag_size > 0) recorder_.own_query(req);
}

void DeepBot::observe_move(const Move& move) {
  recorder_.observe(move);
  ++ply_;
}

uint64_t DeepBot::turn_seed() const {
  return util::splitmix64(params_.seed ^ util::splitmix64((games_ << 16) | uint64_t(ply_)));
}

MoveDecision DeepBot::make_move(const MoveRequest& req) {
  // EndgameAgent solves the turns with the bag empty, so this is its fallback,
  // and every later turn is one too: no query will read the racks not
  // recorded since.
  if (req.bag_size == 0) return hasty_best_move_wmp(req);
  if (!req.opp_rack.empty())
    throw util::Exception("deepbot: hidden leaves only, but the opponent's rack is face up");
  // Every other turn records the rack, which later queries' masks read.
  const belief::LeaveQuery q = recorder_.own_query(req);
  const std::vector<Move> moves = equity_top_k(req, params_.candidates);
  if (moves.size() == 1) return moves.front();
  MoveDecision decision(moves.front());
  decision.values = valued(req, moves, q);
  decision.move = decision.values[best_value(decision.values)].move;
  return decision;
}

std::vector<CandidateValue> DeepBot::valued(const MoveRequest& req, const std::vector<Move>& moves,
                                            const belief::LeaveQuery& q) const {
  const std::vector<double> v = values(req, moves, opp_leaves(q));
  std::vector<CandidateValue> out;
  out.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) out.push_back({moves[i], v[i]});
  return out;
}

std::vector<WeightedLeave> DeepBot::opp_leaves(const belief::LeaveQuery& q) const {
  const std::vector<Rack> draws = opp_draws(q);
  return backend_.values->noisy() ? equal_weights(draws) : weighted_leaves(draws);
}

std::vector<Rack> DeepBot::opp_draws(const belief::LeaveQuery& q) const {
  // Before the opponent's first move, and after a move that kept nothing, the
  // leave is empty, as the leaf's training rows have it. It is still drawn
  // opp_leaves times: a noisy leaf plays each draw out apart.
  if (ply_ == 0 || q.length == 0) return std::vector<Rack>(size_t(params_.opp_leaves));
  // The opponent passed. The leave table holds no 7-tile leaves, so the racks
  // are generated here by other means. As a hack, each is drawn at random from
  // the unseen tiles, as if the opponent had just drawn the whole rack. That
  // ignores what the pass says about the rack and is certainly wrong, but a
  // pass is rare enough in Scrabble that we accept it for now.
  if (q.length > kLeaveArraysMaxLength)
    return random_racks(q.leave_pool, q.length, params_.opp_leaves, turn_seed());
  const std::shared_ptr<const belief::LeaveSampler> b = belief(q);
  // Stratified quantiles, jittered by the turn: each draw stands for an equal
  // share of the belief's mass, so the mean has far less spread than over
  // independent draws.
  const double jitter = double(turn_seed() >> 11) * 0x1.0p-53;
  std::vector<Rack> draws;
  draws.reserve(size_t(params_.opp_leaves));
  for (int i = 0; i < params_.opp_leaves; ++i)
    draws.push_back(b->leave(b->sample((double(i) + jitter) / double(params_.opp_leaves))));
  return draws;
}

std::shared_ptr<const belief::LeaveSampler> DeepBot::belief(const belief::LeaveQuery& q) const {
  if (backend_.predict && q.modelled())
    return belief::beliefs_of({&q, 1}, backend_.predict, *backend_.table)[0];
  // No model's belief (none loaded, or a row its training masks, such as after
  // a pass): the leaves the unseen tiles hold, weighted by their chance.
  return std::make_shared<const belief::LeaveSampler>(belief::LeaveParams{}, q.length, q.leave_pool,
                                                      *backend_.table);
}

std::vector<double> DeepBot::values(const MoveRequest& req, const std::vector<Move>& moves,
                                    const std::vector<WeightedLeave>& opp) const {
  req.board.ensure_movegen_caches(req.dict);
  std::vector<double> out(moves.size(), 0.0);
  const size_t per_call = std::max<size_t>(1, kLeavesPerCall / opp.size());
  for (size_t first = 0; first < moves.size(); first += per_call) {
    const size_t n = std::min(per_call, moves.size() - first);
    value_chunk(req, std::span(moves).subspan(first, n), opp, std::span(out).subspan(first, n));
  }
  return out;
}

void DeepBot::value_chunk(const MoveRequest& req, std::span<const Move> moves,
                          const std::vector<WeightedLeave>& opp, std::span<double> out) const {
  // Each candidate's post-move board is built once (incrementally from the
  // live board's caches) and paired with every opponent leave.
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
  for (size_t i = 0; i < wld.size(); ++i)
    out[i / opp.size()] += opp[i % opp.size()].weight * win_value(wld[i]);
}

}  // namespace scribblez
