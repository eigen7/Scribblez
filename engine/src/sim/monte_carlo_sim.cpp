#include "sim/monte_carlo_sim.h"

#include "agent/agent.h"
#include "agent/endgame_hasty_bot.h"
#include "game/bag.h"
#include "game/game.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "sim/sim_runner.h"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace scribblez {

namespace {

// What one finished rollout contributes: the final score delta (start_player's
// POV) plus the two seats' first moves of the rollout, the placement planes read.
// The opponent moves first, so records[0] is its move and records[1] is
// start_player's; a missing move stays a default Move (PASS), which places nothing.
struct RolloutResult {
  int delta = 0;
  Move opp_first{};   // opponent's first move of the rollout (they move first)
  Move self_first{};  // start_player's first move of the rollout
};

// The leave a rollout seats the opponent with, under the condition. Face-up:
// the known leave, every rollout. Hidden: a draw from the posterior their last
// move induces, or nothing (a uniform full draw) when that move carried no
// information -- which is also what an empty face-up leave amounts to, so the
// two conditions coincide on every position whose opponent kept nothing.
class OppLeaveSampler {
 public:
  OppLeaveSampler(const ParsedGcgPostMove& pos, const Dictionary& dict, LeaveCondition condition,
                  const belief::RackInferrer::Params& infer);

  Rack leave(uint64_t seed) const;

 private:
  // A stream of its own, so the variate that picks the leave is not also the
  // first number that shuffles the pool it is then refilled from.
  static constexpr uint64_t kLeaveStream = 0x9E3779B97F4A7C15ULL;
  Rack fixed_;
  belief::RackPosterior posterior_;
};

OppLeaveSampler::OppLeaveSampler(const ParsedGcgPostMove& pos, const Dictionary& dict,
                                 LeaveCondition condition,
                                 const belief::RackInferrer::Params& infer) {
  if (condition == LeaveCondition::kFaceUp) {
    fixed_ = pos.opp_leave;
  } else if (pos.opp_observation) {
    posterior_ = belief::RackInferrer(dict, infer).infer(*pos.opp_observation, /*seed=*/0);
  }
}

Rack OppLeaveSampler::leave(uint64_t seed) const {
  if (posterior_.empty()) return fixed_;
  std::mt19937_64 rng(seed ^ kLeaveStream);
  return posterior_.sample(std::uniform_real_distribution<double>(0.0, 1.0)(rng));
}

// One rollout (seed g): play to the end from start_player's POV. The unseen pool
// (shared helper in sim_runner.h) is built from the board and start_player's
// leave; the opponent is seated with the leave the sampler gives this rollout
// and draws the rest of its rack from the pool.
RolloutResult rollout(const ParsedGcgPostMove& pos, const Dictionary& dict, Agent& a0, Agent& a1,
                      const OppLeaveSampler& sampler, bool face_up, uint64_t seed) {
  const int opponent = 1 - pos.start_player;  // moved before start_player, so plays first
  std::array<Rack, 2> known;
  known[pos.start_player] = pos.leave;  // start_player keeps its leave
  known[opponent] = sampler.leave(seed);
  Bag pool = unseen_pool(pos.board, pos.leave, seed);
  for (int i = 0; i < known[opponent].size(); ++i) pool.remove(known[opponent].tiles()[i]);
  Game game(a0, a1, dict, seed);
  game.set_face_up_leaves(face_up);
  game.play_from(pos.board, pos.scores, known, pool, /*to_move=*/opponent);
  const GameLog log = game.log();

  RolloutResult r;
  r.delta = log.final_scores[pos.start_player] - log.final_scores[opponent];
  if (log.num_records >= 1 && log.records[0].player == opponent) r.opp_first = log.records[0].move;
  if (log.num_records >= 2 && log.records[1].player == pos.start_player)
    r.self_first = log.records[1].move;
  return r;
}

// Add one seat's placed squares to its marginal `next` plane, and to its `win`
// plane when `won` (that seat strictly won the rollout).
void accumulate_placement(const Move& move, bool won,
                          std::array<int, PlacementCounts::kCells>* next,
                          std::array<int, PlacementCounts::kCells>* win) {
  visit_placed_squares(move, [&](int r, int c) {
    const int cell = r * BOARD_SIZE + c;
    ++(*next)[cell];
    if (won) ++(*win)[cell];
  });
}

// Fold one rollout's outcome into `out`: W/L/D, the exact delta histogram, and
// both seats' placement planes.
void accumulate_rollout(const RolloutResult& r, MonteCarloResult* out) {
  ++out->n;
  if (r.delta > 0)
    ++out->wins;
  else if (r.delta < 0)
    ++out->losses;
  else
    ++out->draws;
  ++out->delta_hist[r.delta];
  accumulate_placement(r.opp_first, r.delta < 0, &out->placement.opp_next, &out->placement.opp_win);
  accumulate_placement(r.self_first, r.delta > 0, &out->placement.self_next,
                       &out->placement.self_win);
}

// Worker: plays games {t+1, t+1+threads, ...} (each seeded by its own g, so the
// thread split doesn't affect any game's outcome) and accumulates into *out.
void monte_carlo_worker(const ParsedGcgPostMove& pos, const Dictionary& dict, int n, int threads,
                        int t, const OppLeaveSampler& sampler, bool face_up,
                        MonteCarloResult* out) {
  // Default solver params, matching the self-play generation that produces the
  // training data (py/scribblez/selfplay.py): the ground truth must reflect the
  // same rollout policy the model's targets are drawn from.
  EndgameHastyBotAgent::Params p0;
  p0.hasty.thread_id = t;
  p0.hasty.name = "H0";
  EndgameHastyBotAgent::Params p1 = p0;
  p1.hasty.name = "H1";
  EndgameHastyBotAgent a0(p0), a1(p1);  // temperature 0 -> deterministic greedy argmax
  for (int g = t + 1; g <= n; g += threads)
    accumulate_rollout(rollout(pos, dict, a0, a1, sampler, face_up, uint64_t(g)), out);
}

// A flat row-major 15x15 count plane as a nested [row][col] JSON array (board
// frame; no model-input flip applied).
boost::json::array plane_to_json(const std::array<int, PlacementCounts::kCells>& plane) {
  boost::json::array rows;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    boost::json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) row.push_back(plane[r * BOARD_SIZE + c]);
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace

boost::json::object MonteCarloResult::to_json() const {
  boost::json::object hist;
  for (const auto& [delta, count] : delta_hist) hist[std::to_string(delta)] = count;
  boost::json::object o;
  o["start_player"] = start_player;
  o["n"] = n;
  o["wld"] = boost::json::object{{"win", wins}, {"loss", losses}, {"draw", draws}};
  o["score_delta_hist"] = std::move(hist);
  // Keyed by the position-evaluation model's placement-head names, so the
  // dashboard pairs each plane with the head it is ground truth for.
  o["placement"] = boost::json::object{{"opp_next_placement", plane_to_json(placement.opp_next)},
                                       {"self_next_placement", plane_to_json(placement.self_next)},
                                       {"opp_win_placement", plane_to_json(placement.opp_win)},
                                       {"self_win_placement", plane_to_json(placement.self_win)}};
  return o;
}

const char* leave_condition_name(LeaveCondition condition) {
  return condition == LeaveCondition::kFaceUp ? "face-up-leaves" : "hidden-leaves";
}

MonteCarloResult run_monte_carlo(const ParsedGcgPostMove& pos, const Dictionary& dict, int n,
                                 int threads, LeaveCondition condition,
                                 const belief::RackInferrer::Params& infer) {
  threads = std::clamp(threads, 1, std::max(1, n));
  const OppLeaveSampler sampler(pos, dict, condition, infer);
  const bool face_up = condition == LeaveCondition::kFaceUp;
  std::vector<MonteCarloResult> partials(threads);
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(monte_carlo_worker, std::cref(pos), std::cref(dict), n, threads, t,
                         std::cref(sampler), face_up, &partials[t]);
  for (auto& w : workers) w.join();

  MonteCarloResult total;
  total.start_player = pos.start_player;
  for (const MonteCarloResult& r : partials) {
    total.n += r.n;
    total.wins += r.wins;
    total.losses += r.losses;
    total.draws += r.draws;
    for (const auto& [delta, count] : r.delta_hist) total.delta_hist[delta] += count;
    for (int i = 0; i < PlacementCounts::kCells; ++i) {
      total.placement.opp_next[i] += r.placement.opp_next[i];
      total.placement.self_next[i] += r.placement.self_next[i];
      total.placement.opp_win[i] += r.placement.opp_win[i];
      total.placement.self_win[i] += r.placement.self_win[i];
    }
  }
  return total;
}

}  // namespace scribblez
