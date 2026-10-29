#include "agent/bob_subtree.h"

#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "sim/pool_labeler.h"
#include "sim/sim_runner.h"
#include "util/work_pool.h"

#include <memory>
#include <optional>
#include <utility>

namespace scribblez {

namespace {

constexpr uint64_t kSubtreeDomain = 0x5B0B5B0B5B0B5B0BULL;
constexpr uint64_t kAfterReplyDomain = 0xA11CEA11CEA11CEULL;

// Bob's belief inputs after `candidate`, one query per rack, and the model's
// parameters for them; both empty without a model. Each rack's sampler is
// built only where it is used, so a turn never holds a candidate's hundred
// table passes at once.
struct BobBeliefs {
  std::vector<belief::LeaveQuery> queries;
  std::vector<belief::LeaveParams> params;

  std::shared_ptr<const belief::LeaveSampler> sampler(size_t rack,
                                                      const LeaveValueArrays& table) const;
};

std::shared_ptr<const belief::LeaveSampler> BobBeliefs::sampler(
  size_t rack, const LeaveValueArrays& table) const {
  if (queries.empty()) return nullptr;
  return belief::belief_of(queries[rack], params[rack], table);
}

BobBeliefs bob_beliefs(const AliceTurn& turn, const Move& candidate, const BeliefModel& model,
                       const Dictionary& dict) {
  if (!model.predict) return {};
  const belief::ReplyView view(turn.position, candidate, dict);
  BobBeliefs out{.queries = std::vector<belief::LeaveQuery>(turn.bob_racks.size()), .params = {}};
  util::parallel_for(turn.bob_racks.size(),
                     [&](size_t i) { out.queries[i] = view.query(turn.bob_racks[i]); });
  out.params = belief::predict_beliefs(out.queries, model.predict);
  return out;
}

// One of Alice's candidates' subtrees: the jobs of the racks where Bob has a
// choice to sim, keyed by rack index, and the replies of the others in place.
struct CandidateSubtrees {
  std::vector<std::pair<size_t, PoolJob>> jobs;
  std::vector<std::vector<Move>> moves;  // Bob's candidates, by rack
};

CandidateSubtrees candidate_subtrees(const AliceTurn& turn, size_t c, const BeliefModel& model,
                                     const SubtreeParams& params, const Dictionary& dict,
                                     uint64_t seed, std::vector<SubtreeMove>& replies) {
  const auto beliefs = bob_beliefs(turn, turn.candidates[c], model, dict);
  const size_t racks = turn.bob_racks.size();
  CandidateSubtrees out{.jobs = {}, .moves = std::vector<std::vector<Move>>(racks)};
  std::vector<std::optional<PoolJob>> jobs(racks);
  util::parallel_for(racks, [&](size_t i) {
    const SimPosition pos = mover_position(turn.applied[c], turn.bob_racks[i]);
    out.moves[i] = mover_candidates(pos, params, dict);
    if (out.moves[i].size() <= 1) {
      replies[i] = {.move = out.moves[i].empty() ? Move::pass() : out.moves[i].front()};
      return;
    }
    const auto belief = beliefs.sampler(i, *model.table);
    jobs[i] = position_job(pos, out.moves[i], rack_seed(seed, kSubtreeDomain, i), belief.get(),
                           dict, params.opp_racks);
  });
  for (size_t i = 0; i < racks; ++i)
    if (jobs[i]) out.jobs.emplace_back(i, std::move(*jobs[i]));
  return out;
}

}  // namespace

std::vector<Rack> bob_racks(const RolloutGroup& group, const RolloutStart& start, int count) {
  RolloutBatch batch{.groups = {group}, .starts = {start}, .tasks = {}};
  batch.starts[0].group = 0;
  std::vector<Rack> out;
  out.reserve(size_t(count));
  for (int i = 0; i < count; ++i)
    out.push_back(rollout_racks(batch, {.position = 0, .rollout = uint32_t(i)}).opp);
  return out;
}

std::vector<std::vector<SubtreeMove>> bob_replies(const AliceTurn& turn,
                                                  const std::vector<size_t>& which,
                                                  const BeliefModel& model, RolloutValues& values,
                                                  const SubtreeParams& params,
                                                  const Dictionary& dict, uint64_t seed) {
  std::vector<std::vector<SubtreeMove>> replies(turn.candidates.size());
  for (size_t c : which) replies[c].resize(turn.bob_racks.size());
  // Bob's candidates by (candidate, rack), kept until the labels come back.
  std::vector<std::vector<std::vector<Move>>> moves(turn.candidates.size());
  ChunkedJobs jobs(which.size(), [&](size_t chunk) {
    const size_t c = which[chunk];
    CandidateSubtrees made = candidate_subtrees(turn, c, model, params, dict, seed, replies[c]);
    moves[c] = std::move(made.moves);
    for (auto& [i, job] : made.jobs) i += c * turn.bob_racks.size();
    return std::move(made.jobs);
  });
  label_all(values, {.rollouts = params.opp_racks, .race = params.race}, jobs,
            [&](size_t key, const PoolPosition& p) {
              const size_t c = key / turn.bob_racks.size();
              const size_t i = key % turn.bob_racks.size();
              replies[c][i] = chosen_move(moves[c][i], p);
            });
  return replies;
}

std::vector<double> values_after_replies(const AliceTurn& turn, const std::vector<size_t>& which,
                                         const std::vector<std::vector<SubtreeMove>>& replies,
                                         RolloutValues& values, int sims, const Dictionary& dict,
                                         uint64_t seed) {
  const size_t racks = turn.bob_racks.size();
  std::vector<double> out(turn.candidates.size(), kPruned);
  for (size_t c : which) out[c] = 0.0;
  ChunkedJobs jobs(which.size(), [&](size_t chunk) {
    const size_t c = which[chunk];
    std::vector<std::pair<size_t, PoolJob>> made(racks);
    util::parallel_for(racks, [&](size_t i) {
      made[i] = {c * racks + i,
                 after_reply_job(turn.applied[c], turn.bob_racks[i], replies[c][i].move, sims, dict,
                                 rack_seed(seed, kAfterReplyDomain, i))};
    });
    return made;
  });
  // The leaf values each sim from Bob's side, who moved last.
  label_all(values, {.rollouts = sims, .race = false}, jobs,
            [&](size_t key, const PoolPosition& p) {
              out[key / racks] += (1.0 - p.labels[0].win_prob) / double(racks);
            });
  return out;
}

}  // namespace scribblez
