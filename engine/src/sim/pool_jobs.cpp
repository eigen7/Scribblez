#include "sim/pool_jobs.h"

#include "belief/opp_rack_summary.h"
#include "data/forgone_play_log.h"
#include "lexicon/dictionary.h"
#include "sim/hasty_rollout_task.h"
#include "util/exception.h"

#include <memory>
#include <thread>

namespace scribblez {

std::vector<Rack> belief_leaves(const belief::LeaveSampler& belief, uint64_t seed, int rollouts) {
  std::vector<Rack> out;
  out.reserve(size_t(rollouts));
  for (int i = 0; i < rollouts; ++i)
    out.push_back(belief.leave(belief.sample(belief::rollout_leave_quantile(seed + uint64_t(i)))));
  return out;
}

PoolJob make_pool_job(const SimmedPosition& r, const belief::LeaveSampler* leave,
                      const Dictionary& dict, int rollouts) {
  const SimPosition& p = r.position;
  PoolJob job;
  job.group = make_rollout_group(p.board, p.rack, r.base_seed);
  if (leave) job.group.opp_leaves = belief_leaves(*leave, r.base_seed, rollouts);
  job.record.at = r.pos;
  job.record.num_legal_moves = r.candidates.num_legal_moves;
  job.record.moves = r.candidates.moves;
  job.record.equity_ranks = r.candidates.equity_ranks;
  job.record.opp_rack = belief::summarize_opp_rack(job.group.pool, leave);
  Board board = p.board;
  board.ensure_movegen_caches(dict);
  for (const Move& m : r.candidates.moves) {
    job.starts.push_back(
      make_rollout_start(board, p.rack, p.scores[p.mover], p.scores[1 - p.mover], m, /*group=*/0));
  }
  return job;
}

std::vector<PoolJob> make_pool_jobs(const std::vector<SimmedPosition>& selected,
                                    const belief::SlogLeaveBeliefs* beliefs, const Dictionary& dict,
                                    int rollouts, int threads) {
  std::vector<PoolJob> jobs(selected.size());
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      for (size_t i = size_t(t); i < selected.size(); i += size_t(threads)) {
        const std::shared_ptr<const belief::LeaveSampler> leave =
          beliefs ? beliefs->belief(selected[i].pos) : nullptr;
        jobs[i] = make_pool_job(selected[i], leave.get(), dict, rollouts);
      }
    });
  }
  for (std::thread& w : workers) w.join();
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
