#include "agent/subtree_search.h"

#include "agent/agent.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "util/assert.h"
#include "util/math.h"

#include <algorithm>
#include <cmath>
#include <numeric>

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

std::vector<SubtreeMove> contenders(const std::vector<Move>& moves, const PoolPosition& labeled,
                                    int sims) {
  const std::vector<PoolLabel>& labels = labeled.labels;
  std::vector<size_t> order(moves.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return labels[a].q > labels[b].q; });
  std::vector<double> q;
  for (const size_t k : order) q.push_back(labels[k].q);
  const std::vector<int> share = util::apportion(q, sims);
  std::vector<SubtreeMove> out;
  double total = 0;
  // Every move after the first without sims has none either.
  for (size_t j = 0; j < order.size() && share[j] > 0; ++j) {
    const size_t k = order[j];
    out.push_back({.move = moves[k],
                   .equity_rank = int(k),
                   .candidates = int(moves.size()),
                   .q = labels[k].q,
                   .sims = share[j]});
    total += labels[k].q;
  }
  for (SubtreeMove& m : out) m.weight = m.q / total;
  return out;
}

SubtreeMove chosen_move(const std::vector<Move>& moves, const PoolPosition& labeled) {
  return contenders(moves, labeled, /*sims=*/1).front();
}

PoolJob after_reply_job(const RolloutStart& applied, const Rack& rack,
                        const std::vector<Move>& replies, const std::vector<int>& sims,
                        const Dictionary& dict, uint64_t seed) {
  const int most = *std::max_element(sims.begin(), sims.end());
  PoolJob job = position_job(mover_position(applied, rack), replies, seed, nullptr, dict, most);
  job.rollouts = sims;
  // The replier's unseen pool less the leave, which the sims seat, and the
  // exchanged tiles, which went back only after the refill: the bag as the
  // mover of `applied` found it.
  const bool unseen = job.group.pool.remove(applied.returned.counts());
  DEBUG_ASSERT(unseen, "an exchange's tiles are unseen to the replier");
  job.group.opp_leaves.assign(size_t(most), applied.leave);
  for (RolloutStart& start : job.starts) start.opp_returned = applied.returned;
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

long long label_all(RolloutValues& values, const PoolLabeler::Params& params, ChunkedJobs& jobs,
                    const std::function<void(size_t, const PoolPosition&)>& done) {
  PoolLabeler labeler(values, params);
  size_t emitted = 0;
  labeler.run([&] { return jobs.next(); },
              [&](PoolPosition&& p) { done(jobs.keys()[emitted++], p); });
  return labeler.stats().tasks;
}

}  // namespace scribblez
