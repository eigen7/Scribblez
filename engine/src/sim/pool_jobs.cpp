#include "sim/pool_jobs.h"

#include "data/forgone_play_log.h"
#include "lexicon/dictionary.h"
#include "sim/hasty_rollout_task.h"
#include "training/policy_set_inputs.h"
#include "util/exception.h"
#include "util/work_pool.h"

#include <memory>

namespace scribblez {

std::vector<Rack> belief_leaves(const belief::LeaveSampler& belief, uint64_t seed, int rollouts) {
  std::vector<Rack> out;
  out.reserve(size_t(rollouts));
  for (int i = 0; i < rollouts; ++i)
    out.push_back(belief.leave(belief.sample(belief::rollout_leave_quantile(seed + uint64_t(i)))));
  return out;
}

PoolJob position_job(const SimPosition& pos, const std::vector<Move>& moves, uint64_t seed,
                     const belief::LeaveSampler* leave, const Dictionary& dict, int rollouts) {
  PoolJob job;
  job.group = make_rollout_group(pos.board, pos.rack, seed);
  if (leave) job.group.opp_leaves = belief_leaves(*leave, seed, rollouts);
  job.record.moves = moves;
  Board board = pos.board;
  board.ensure_movegen_caches(dict);
  for (const Move& m : moves) {
    job.starts.push_back(make_rollout_start(board, pos.rack, pos.scores[pos.mover],
                                            pos.scores[1 - pos.mover], m, /*group=*/0));
  }
  return job;
}

PoolJob make_pool_job(const SimmedPosition& r, const belief::LeaveSampler* leave,
                      const Dictionary& dict, int rollouts) {
  PoolJob job = position_job(r.position, r.candidates.moves, r.base_seed, leave, dict, rollouts);
  job.record.at = r.pos;
  job.record.num_legal_moves = r.candidates.num_legal_moves;
  job.record.equity_ranks = r.candidates.equity_ranks;
  job.record.opp_rack = policy_set::opp_rack_summary(r.position.board, r.position.rack, leave);
  return job;
}

std::vector<PoolJob> make_pool_jobs(const std::vector<SimmedPosition>& selected,
                                    const belief::SlogLeaveBeliefs* beliefs, const Dictionary& dict,
                                    int rollouts) {
  std::vector<PoolJob> jobs(selected.size());
  util::parallel_for(selected.size(), [&](size_t i) {
    const std::shared_ptr<const belief::LeaveSampler> leave =
      beliefs ? beliefs->belief(selected[i].pos) : nullptr;
    jobs[i] = make_pool_job(selected[i], leave.get(), dict, rollouts);
  });
  return jobs;
}

void check_forgone_sidecars(const std::vector<std::filesystem::path>& slogs) {
  for (const std::filesystem::path& slog : slogs) {
    if (!std::filesystem::exists(forgone_path_of(slog.string())))
      throw util::CleanException("{} has no .forgone sidecar; run forgone_play_tool over it first",
                                 slog.string());
  }
}

}  // namespace scribblez
