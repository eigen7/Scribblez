#include "agent/alice_subtree.h"

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "util/assert.h"
#include "util/work_pool.h"

#include <memory>
#include <optional>
#include <utility>

namespace scribblez {

namespace {

constexpr uint64_t kSecondTurnDomain = 0x2A11CE2A11CE2A11ULL;
constexpr uint64_t kAfterSecondDomain = 0xA2A2A11CEA2A2A2AULL;

SecondTurn second_turn(const RolloutBatch& batch, uint32_t rollout, const Dictionary& dict) {
  const RolloutStart& a = batch.starts[0];
  const RolloutRacks racks = rollout_racks(batch, {.position = 0, .rollout = rollout});
  DEBUG_ASSERT(racks.bag > 0, "a searched candidate leaves tiles in the bag");
  // HastyBot's request, as the rollouts make it (hasty_rollout_task.cpp).
  const MoveRequest req{a.board, dict, racks.opp, Rack{}, a.opp_score, a.my_score, racks.bag};
  const Move reply = hasty_best_move_wmp(req);
  return {.bob_rack = racks.opp,
          .bag = racks.bag,
          .reply = reply,
          .bob = make_rollout_start(a.board, racks.opp, a.opp_score, a.my_score, reply, 0),
          .alice_rack = racks.mine};
}

// Alice to move at `t`: the position her sims start from.
SimPosition alice_position(const SecondTurn& t) { return mover_position(t.bob, t.alice_rack); }

// Alice's belief inputs at a candidate's second turns where Bob's leave is
// hidden from her (a bag left after his draw), and the model's parameters for
// them; none without a model. Each sampler is built only where it is used, as
// in Bob's subtree.
struct SecondTurnBeliefs {
  std::vector<std::optional<belief::LeaveQuery>> queries;  // by rack
  std::vector<belief::LeaveParams> params;                 // by query, in rack order
  std::vector<size_t> param_index;                         // by rack

  std::shared_ptr<const belief::LeaveSampler> sampler(size_t rack,
                                                      const LeaveValueArrays& table) const;
};

std::shared_ptr<const belief::LeaveSampler> SecondTurnBeliefs::sampler(
  size_t rack, const LeaveValueArrays& table) const {
  if (queries.empty() || !queries[rack]) return nullptr;
  return belief::belief_of(*queries[rack], params[param_index[rack]], table);
}

// Alice's query at `t`: Bob, the chooser, replied from the position after
// her candidate `a`.
belief::LeaveQuery second_turn_query(const RolloutStart& a, const SecondTurn& t,
                                     std::span<const belief::ChooserMove> bob_earlier,
                                     const Dictionary& dict) {
  const belief::ReplyPosition bob{.board = a.board,
                                  .chooser_rack = t.bob_rack,
                                  .chooser_score = a.opp_score,
                                  .opp_score = a.my_score,
                                  .bag = t.bag,
                                  .earlier = bob_earlier};
  return belief::ReplyView(bob, t.reply, dict).query(t.alice_rack);
}

SecondTurnBeliefs second_turn_beliefs(const RolloutStart& a, const std::vector<SecondTurn>& turns,
                                      std::span<const belief::ChooserMove> bob_earlier,
                                      const BeliefModel& model, const Dictionary& dict) {
  if (!model.predict) return {};
  SecondTurnBeliefs out{.queries = std::vector<std::optional<belief::LeaveQuery>>(turns.size()),
                        .params = {},
                        .param_index = std::vector<size_t>(turns.size())};
  util::parallel_for(turns.size(), [&](size_t i) {
    const SimPosition pos = alice_position(turns[i]);
    if (pos.board.pov_bag_size(pos.rack.size()) > 0)
      out.queries[i] = second_turn_query(a, turns[i], bob_earlier, dict);
  });
  std::vector<belief::LeaveQuery> asked;
  for (size_t i = 0; i < turns.size(); ++i) {
    out.param_index[i] = asked.size();
    if (out.queries[i]) asked.push_back(*out.queries[i]);
  }
  out.params = belief::predict_beliefs(asked, model.predict);
  return out;
}

// One candidate's second turns: the jobs of the racks where Alice has a choice
// to sim, keyed by rack index; the moves of the others go in place.
std::vector<std::pair<size_t, PoolJob>> second_turn_jobs(
  const RolloutStart& a, const std::vector<SecondTurn>& turns,
  std::span<const belief::ChooserMove> bob_earlier, const BeliefModel& model,
  const SubtreeParams& params, const Dictionary& dict, uint64_t seed,
  std::vector<std::vector<Move>>& moves, std::vector<Move>& chosen) {
  const SecondTurnBeliefs beliefs = second_turn_beliefs(a, turns, bob_earlier, model, dict);
  std::vector<std::optional<PoolJob>> jobs(turns.size());
  util::parallel_for(turns.size(), [&](size_t i) {
    const SimPosition pos = alice_position(turns[i]);
    moves[i] = mover_candidates(pos, params, dict);
    if (moves[i].size() <= 1) {
      chosen[i] = moves[i].empty() ? Move::pass() : moves[i].front();
      return;
    }
    const auto belief = beliefs.sampler(i, *model.table);
    jobs[i] = position_job(pos, moves[i], rack_seed(seed, kSecondTurnDomain, i), belief.get(), dict,
                           params.opp_racks);
  });
  std::vector<std::pair<size_t, PoolJob>> out;
  for (size_t i = 0; i < turns.size(); ++i)
    if (jobs[i]) out.emplace_back(i, std::move(*jobs[i]));
  return out;
}

// Alice's second move at each rack of each candidate in `which`, and the
// turns it was chosen at; the other candidates' rows stay empty.
struct SecondMoves {
  std::vector<std::vector<SecondTurn>> turns;
  std::vector<std::vector<Move>> chosen;
};

SecondMoves second_moves(const AliceTurn& turn, const std::vector<size_t>& which,
                         std::span<const belief::ChooserMove> bob_earlier, const BeliefModel& model,
                         RolloutValues& values, const SubtreeParams& params, const Dictionary& dict,
                         uint64_t seed) {
  const size_t racks = turn.bob_racks.size();
  SecondMoves out{.turns = std::vector<std::vector<SecondTurn>>(turn.candidates.size()),
                  .chosen = std::vector<std::vector<Move>>(turn.candidates.size())};
  // Alice's candidates by (candidate, rack), kept until the labels come back.
  std::vector<std::vector<std::vector<Move>>> moves(turn.candidates.size());
  ChunkedJobs jobs(which.size(), [&](size_t chunk) {
    const size_t c = which[chunk];
    out.turns[c] = second_turns(turn, c, dict);
    moves[c].resize(racks);
    out.chosen[c].resize(racks);
    auto made = second_turn_jobs(turn.applied[c], out.turns[c], bob_earlier, model, params, dict,
                                 seed, moves[c], out.chosen[c]);
    for (auto& [i, job] : made) i += c * racks;
    return made;
  });
  label_all(values, {.rollouts = params.opp_racks, .race = params.race}, jobs,
            [&](size_t key, const PoolPosition& p) {
              const size_t c = key / racks;
              const size_t i = key % racks;
              out.chosen[c][i] = chosen_move(moves[c][i], p).move;
            });
  return out;
}

// The mean over racks of Alice's win probability in `sims` rollouts after her
// second move, for each candidate in `which`; kPruned for the rest.
std::vector<double> values_after_second_moves(const SecondMoves& second,
                                              const std::vector<size_t>& which,
                                              RolloutValues& values, int sims,
                                              const Dictionary& dict, uint64_t seed) {
  std::vector<double> out(second.turns.size(), kPruned);
  if (which.empty()) return out;
  const size_t racks = second.turns[which.front()].size();
  for (size_t c : which) out[c] = 0.0;
  ChunkedJobs jobs(which.size(), [&](size_t chunk) {
    const size_t c = which[chunk];
    std::vector<std::pair<size_t, PoolJob>> made(racks);
    util::parallel_for(racks, [&](size_t i) {
      const SecondTurn& t = second.turns[c][i];
      made[i] = {c * racks + i, after_reply_job(t.bob, t.alice_rack, second.chosen[c][i], sims,
                                                dict, rack_seed(seed, kAfterSecondDomain, i))};
    });
    return made;
  });
  // Alice is the mover of these sims: the leaf values them from her side.
  label_all(values, {.rollouts = sims, .race = false}, jobs,
            [&](size_t key, const PoolPosition& p) {
              out[key / racks] += p.labels[0].win_prob / double(racks);
            });
  return out;
}

}  // namespace

std::vector<SecondTurn> second_turns(const AliceTurn& turn, size_t c, const Dictionary& dict) {
  RolloutBatch batch{.groups = {turn.group}, .starts = {turn.applied[c]}, .tasks = {}};
  batch.starts[0].group = 0;
  std::vector<SecondTurn> out(turn.bob_racks.size());
  util::parallel_for(out.size(), [&](size_t i) { out[i] = second_turn(batch, uint32_t(i), dict); });
  return out;
}

std::vector<double> alice_subtree_values(const AliceTurn& turn, const std::vector<size_t>& which,
                                         std::span<const belief::ChooserMove> bob_earlier,
                                         const BeliefModel& model, RolloutValues& values,
                                         const SubtreeParams& params, int sims,
                                         const Dictionary& dict, uint64_t seed) {
  const SecondMoves second =
    second_moves(turn, which, bob_earlier, model, values, params, dict, seed);
  return values_after_second_moves(second, which, values, sims, dict, seed);
}

}  // namespace scribblez
