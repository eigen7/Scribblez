#include "sim/grouped_pools.h"

#include "agent/agent.h"
#include "agent/bob_subtree.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "training/policy_set_inputs.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/work_pool.h"

#include <limits>
#include <utility>

namespace scribblez {

namespace {

constexpr uint64_t kGroupRootDomain = 0x6B0B6B0B6B0B6B0BULL;
constexpr uint64_t kGroupPoolDomain = 0x9B0B9B0B9B0B9B0BULL;

// Every legal move of `pos`'s mover, best static equity first, from the
// mover's own information.
std::vector<Move> ranked_moves(const SimPosition& pos, const Dictionary& dict) {
  const int bag = pos.board.pov_bag_size(pos.rack.size());
  const MoveRequest req{pos.board, dict, pos.rack, Rack{}, pos.scores[0], pos.scores[1], bag};
  return equity_top_k(req, std::numeric_limits<int>::max());
}

}  // namespace

void GroupParams::validate() const {
  if (moves < 1) throw util::CleanException("--group-moves must be >= 1");
  if (racks < 1) throw util::CleanException("--group-racks must be >= 1");
  if (map_sims < 1 || map_sims > rollouts)
    throw util::CleanException("--group-map-sims must be in [1, --rollouts]");
  if (map_steps < 1 || map_steps > map_sims)
    throw util::CleanException("--group-map-steps must be in [1, --group-map-sims]");
}

GroupBuild::GroupBuild(GroupParent parent, const BeliefModel& bob_model, const GroupParams& params,
                       const Dictionary& dict, uint64_t seed)
    : parent_(std::move(parent)), bob_model_(bob_model), params_(params), dict_(dict), seed_(seed) {
  // Alice's rollout i seats rack i for Bob, in the group her moves share.
  const PoolJob root = position_job(parent_.position, parent_.moves, seed_ ^ kGroupRootDomain,
                                    parent_.belief.get(), dict_, params_.racks);
  record_.at = parent_.at;
  record_.moves = parent_.moves;
  record_.racks = other_racks(root.group, root.starts[0], params_.racks);
  applied_ = root.starts;
}

PoolJob GroupBuild::pool_job(size_t move, size_t rack, const belief::LeaveSampler* belief) const {
  const SimPosition pos = mover_position(applied_[move], record_.racks[rack]);
  const std::vector<Move> ranked = ranked_moves(pos, dict_);
  policy_set::PoolCandidates cands =
    policy_set::pool_candidates(ranked, params_.blank_prune_points);
  PoolJob job = position_job(pos, cands.moves, rack_seed(seed_, kGroupPoolDomain, rack), belief,
                             dict_, params_.rollouts);
  job.record.at = parent_.at;
  job.record.num_legal_moves = uint32_t(ranked.size());
  job.record.equity_ranks = std::move(cands.equity_ranks);
  job.record.opp_rack = policy_set::opp_rack_summary(pos.board, pos.rack, belief);
  return job;
}

std::vector<PoolJob> GroupBuild::pool_jobs() {
  const size_t moves = record_.moves.size();
  const size_t racks = record_.racks.size();
  const SimPosition& pos = parent_.position;
  const belief::ReplyPosition position{.board = pos.board,
                                       .chooser_rack = pos.rack,
                                       .chooser_score = pos.scores[size_t(pos.mover)],
                                       .opp_score = pos.scores[size_t(1 - pos.mover)],
                                       .bag = parent_.bag,
                                       .earlier = parent_.earlier};
  std::vector<PoolJob> jobs(moves * racks);
  for (size_t a = 0; a < moves; ++a) {
    const BobBeliefs beliefs =
      bob_beliefs(position, record_.moves[a], record_.racks, bob_model_, dict_);
    util::parallel_for(racks, [&](size_t r) {
      const auto belief = beliefs.sampler(r, *bob_model_.table);
      jobs[a * racks + r] = pool_job(a, r, belief.get());
    });
  }
  record_.pools.resize(jobs.size());
  pool_groups_.clear();
  for (const PoolJob& job : jobs) pool_groups_.push_back(job.group);
  return jobs;
}

void GroupBuild::pool_labeled(size_t index, PoolPosition&& p) {
  record_.pools[index] = std::move(p);
}

PoolJob GroupBuild::map_job(size_t index, std::vector<MapKey>* keys) const {
  const PoolPosition& p = record_.pools[index];
  const size_t racks = record_.racks.size();
  const SimPosition pos = mover_position(applied_[index / racks], record_.racks[index % racks]);
  const std::vector<int> counts = map_prefixes(params_.map_sims, params_.map_steps);
  std::vector<Move> moves;
  std::vector<int> rollouts;
  for (const SubtreeMove& m : contenders(p.moves, p, params_.map_sims)) {
    for (const int n : counts) {
      moves.push_back(m.move);
      rollouts.push_back(n);
      keys->push_back({index, m.equity_rank, n});
    }
  }
  PoolJob job = position_job(pos, moves, 0, nullptr, dict_, params_.map_sims);
  job.group = pool_groups_[index];
  job.rollouts = std::move(rollouts);
  return job;
}

std::vector<PoolJob> GroupBuild::map_jobs() {
  std::vector<PoolJob> jobs(record_.pools.size());
  map_keys_.assign(jobs.size(), {});
  util::parallel_for(jobs.size(), [&](size_t i) { jobs[i] = map_job(i, &map_keys_[i]); });
  return jobs;
}

void GroupBuild::map_labeled(size_t index, const PoolPosition& p) {
  const std::vector<MapKey>& keys = map_keys_[index];
  const size_t racks = record_.racks.size();
  RELEASE_ASSERT(p.maps.size() == keys.size());
  for (size_t s = 0; s < keys.size(); ++s) {
    const MapKey& k = keys[s];
    RELEASE_ASSERT(p.maps[s].sims == float(k.sims), "a map of {} sims summed {}", k.sims,
                   p.maps[s].sims);
    record_.maps.push_back({.move = int(k.pool / racks),
                            .rack = int(k.pool % racks),
                            .reply = k.reply,
                            .sims = k.sims,
                            .maps = p.maps[s]});
  }
}

}  // namespace scribblez
