#pragma once

// Grouped pools (docs/plans/deep_bot.md, "Training data: grouped pools") on
// the pool labeler: a parent position with Alice to move, her candidates
// there (GroupParams::quotas), racks sampled for Bob from her belief, and at each (move,
// rack) Bob's pool, labeled as a .pool position is, from his side with
// Alice's leave drawn from his belief after the move; then, for each rack's
// leading replies, the maps of the pool's first few rollouts (data/group_log.h
// holds the record). A group is a search turn's shape, so it is built with
// the subtree's helpers (agent/subtree_search.h).
//
// A group goes through a labeler twice. Its pools race with the file's other
// pools. Once they are labeled, their leading replies are known, and the maps
// of those replies run in an unraced labeler that sums maps
// (PoolLabeler::Params::sum_maps): the maps' cost is paid by every launch
// that sums them, and the maps' rollouts are few, so they get launches of
// their own. A map's rollouts are the pool's own first n, rerun: a task's
// outcome depends only on its start and rollout index.

#include "agent/subtree_search.h"
#include "belief/leave_sampler.h"
#include "belief/reply_query.h"
#include "data/group_log.h"
#include "data/slog_sampling.h"
#include "game/move.h"
#include "sim/pool_labeler.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_candidates.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace scribblez {

class Dictionary;

struct GroupParams {
  // Alice's candidates: the move the game played, then these strata of her
  // static-equity ranking (move_set_eval::stratified_candidates; the head
  // counts beyond the played move, so a group is usually 8). The head is
  // what a v0 search proposes; the draws below it put candidates that differ
  // in mechanism beside it, so that the store the network trains on holds
  // the kind its steering may admit.
  move_set_eval::StratumQuotas quotas{
    .top = 3, .mid = 2, .tail = 1, .exchange = 1, .mid_rank_limit = 32};
  int racks = 8;       // sampled for Bob, the same under every move
  int map_sims = 12;   // a search's sims per rack: the largest map count
  int map_steps = 3;   // the maps' counts, map_prefixes(map_sims, map_steps)
  int rollouts = 100;  // the pools' budget per reply
  int blank_prune_points = 40;

  // The most moves a group can hold: the played move and every quota.
  int max_moves() const { return 1 + quotas.top + quotas.mid + quotas.tail + quotas.exchange; }
  // Throws util::CleanException naming the first bad field.
  void validate() const;
};

// A group's parent: Alice's position, and what the beliefs read.
struct GroupParent {
  binlog::GamePositionIndex at;
  SimPosition position;  // Alice to move, hidden leaves
  int bag = 0;
  std::vector<Move> moves;            // her candidates, best equity first
  std::vector<int32_t> equity_ranks;  // parallel to `moves` (SimCandidates::equity_ranks)
  // Hers about Bob's leave; null draws his whole rack from the unseen pool.
  std::shared_ptr<const belief::LeaveSampler> belief;
  // Her earlier moves, which Bob's belief about her leave reads
  // (belief::ReplyPosition::earlier); unused without a model.
  std::vector<belief::ChooserMove> earlier;
};

// One group through both stages. pool_jobs() hands out its pools, move-major,
// and pool_labeled(i, record) takes each back by that index; map_jobs() then
// hands out the maps' jobs, and map_labeled(j, record) takes each back; the
// record is then complete.
class GroupBuild {
 public:
  GroupBuild(GroupParent parent, const BeliefModel& bob_model, const GroupParams& params,
             const Dictionary& dict, uint64_t seed);

  std::vector<PoolJob> pool_jobs();
  void pool_labeled(size_t index, PoolPosition&& p);
  std::vector<PoolJob> map_jobs();
  void map_labeled(size_t index, const PoolPosition& p);

  const GroupRecord& record() const { return record_; }
  size_t num_pools() const { return record_.moves.size() * record_.racks.size(); }

 private:
  // Which (pool, reply, rollouts) each start of the maps' jobs is, by job.
  struct MapKey {
    size_t pool;
    int reply;
    int sims;
  };

  // Bob's pool at (move, rack) with his belief about Alice's leave there.
  PoolJob pool_job(size_t move, size_t rack, const belief::LeaveSampler* belief) const;
  // The maps' job of pool `index`, the leading replies' starts at each count.
  PoolJob map_job(size_t index, std::vector<MapKey>* keys) const;

  GroupParent parent_;
  const BeliefModel& bob_model_;
  GroupParams params_;
  const Dictionary& dict_;
  uint64_t seed_;
  GroupRecord record_;
  std::vector<RolloutStart> applied_;          // Alice's moves applied, by move
  std::vector<RolloutGroup> pool_groups_;      // each pool's group, kept for its maps
  std::vector<std::vector<MapKey>> map_keys_;  // by maps' job
};

}  // namespace scribblez
