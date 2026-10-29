#include "agent/subtree_search.h"

#include "agent/agent.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "util/assert.h"
#include "util/math.h"

#include <algorithm>
#include <cmath>

namespace scribblez {

size_t best_candidate(const std::vector<double>& values) {
  size_t best = values.size();
  for (size_t c = 0; c < values.size(); ++c)
    if (!std::isnan(values[c]) && (best == values.size() || values[c] > values[best])) best = c;
  return best;
}

uint64_t rack_seed(uint64_t seed, uint64_t domain, size_t rack) {
  return util::splitmix64(seed ^ domain ^ util::splitmix64(uint64_t(rack)));
}

SimPosition mover_position(const RolloutStart& applied, const Rack& rack) {
  return {.board = applied.board,
          .scores = {applied.opp_score, applied.my_score},
          .mover = 0,
          .rack = rack,
          .opp_leave = {},
          .opp_leave_belief = nullptr};
}

std::vector<Move> mover_candidates(const SimPosition& pos, const SubtreeParams& params,
                                   const Dictionary& dict) {
  const Rack hidden;
  const int bag = pos.board.pov_bag_size(pos.rack.size());
  const MoveRequest req{pos.board, dict, pos.rack, hidden, pos.scores[0], pos.scores[1], bag};
  return equity_top_k(req, bag <= params.late_bag ? params.late_candidates : params.candidates);
}

SubtreeMove chosen_move(const std::vector<Move>& moves, const PoolPosition& labeled) {
  const std::vector<PoolLabel>& labels = labeled.labels;
  const auto best = std::max_element(
    labels.begin(), labels.end(), [](const PoolLabel& a, const PoolLabel& b) { return a.q < b.q; });
  const size_t k = size_t(best - labels.begin());
  return {.move = moves[k], .equity_rank = int(k), .candidates = int(moves.size()), .q = best->q};
}

PoolJob after_reply_job(const RolloutStart& applied, const Rack& rack, const Move& reply, int sims,
                        const Dictionary& dict, uint64_t seed) {
  PoolJob job = position_job(mover_position(applied, rack), {reply}, seed, nullptr, dict, sims);
  // The replier's unseen pool less the leave, which the sims seat, and the
  // exchanged tiles, which went back only after the refill: the bag as the
  // mover of `applied` found it.
  const bool unseen = job.group.pool.remove(applied.returned.counts());
  DEBUG_ASSERT(unseen, "an exchange's tiles are unseen to the replier");
  job.group.opp_leaves.assign(size_t(sims), applied.leave);
  job.starts[0].opp_returned = applied.returned;
  return job;
}

std::optional<PoolJob> ChunkedJobs::next() {
  while (buffered_.empty() && chunk_ < chunks_) {
    for (auto& job : make_(chunk_++)) buffered_.push_back(std::move(job));
  }
  if (buffered_.empty()) return std::nullopt;
  auto [key, job] = std::move(buffered_.front());
  buffered_.pop_front();
  keys_.push_back(key);
  return std::move(job);
}

void label_all(RolloutValues& values, const PoolLabeler::Params& params, ChunkedJobs& jobs,
               const std::function<void(size_t, const PoolPosition&)>& done) {
  PoolLabeler labeler(values, params);
  size_t emitted = 0;
  labeler.run([&] { return jobs.next(); },
              [&](PoolPosition&& p) { done(jobs.keys()[emitted++], p); });
}

}  // namespace scribblez
