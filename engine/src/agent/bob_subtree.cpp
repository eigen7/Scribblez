#include "agent/bob_subtree.h"

#include "agent/agent.h"
#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "game/bag.h"
#include "lexicon/dictionary.h"
#include "sim/pool_jobs.h"
#include "sim/pool_labeler.h"
#include "sim/sim_runner.h"
#include "util/assert.h"
#include "util/math.h"
#include "util/work_pool.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace scribblez {

namespace {

// Seeds by rack index, one domain per use, so every candidate of Alice's meets
// rack i's samples and draws: her candidates are compared on paired samples.
constexpr uint64_t kSubtreeDomain = 0x5B0B5B0B5B0B5B0BULL;
constexpr uint64_t kAfterReplyDomain = 0xA11CEA11CEA11CEULL;

uint64_t rack_seed(uint64_t seed, uint64_t domain, size_t rack) {
  return util::splitmix64(seed ^ domain ^ util::splitmix64(uint64_t(rack)));
}

// Bob to move after Alice's candidate, holding `rack`; player 0 is Bob.
SimPosition bob_position(const RolloutStart& applied, const Rack& rack) {
  return {.board = applied.board,
          .scores = {applied.opp_score, applied.my_score},
          .mover = 0,
          .rack = rack,
          .opp_leave = {},
          .opp_leave_belief = nullptr};
}

// Bob's static-equity top moves at `pos`, N by the bag he sees.
std::vector<Move> bob_candidates(const SimPosition& pos, const SubtreeParams& params,
                                 const Dictionary& dict) {
  const Rack hidden;
  const int bag = pos.board.pov_bag_size(pos.rack.size());
  const MoveRequest req{pos.board, dict, pos.rack, hidden, pos.scores[0], pos.scores[1], bag};
  return equity_top_k(req, bag <= params.late_bag ? params.late_candidates : params.candidates);
}

// Bob's beliefs about Alice's leave after `candidate`, one per rack.
std::vector<std::shared_ptr<const belief::LeaveSampler>> bob_beliefs(const AliceTurn& turn,
                                                                     const Move& candidate,
                                                                     const BeliefModel& model,
                                                                     const Dictionary& dict) {
  if (!model.predict)
    return std::vector<std::shared_ptr<const belief::LeaveSampler>>(turn.bob_racks.size());
  const belief::ReplyView view(turn.position, candidate, dict);
  std::vector<belief::LeaveQuery> queries;
  queries.reserve(turn.bob_racks.size());
  for (const Rack& rack : turn.bob_racks) queries.push_back(view.query(rack));
  return belief::beliefs_of(queries, model.predict, *model.table);
}

// Hands a labeler jobs built a chunk at a time (one of Alice's candidates),
// remembering the order they went out in.
class ChunkedJobs {
 public:
  using Maker = std::function<std::vector<std::pair<size_t, PoolJob>>(size_t chunk)>;

  ChunkedJobs(size_t chunks, Maker make) : chunks_(chunks), make_(std::move(make)) {}

  std::optional<PoolJob> next();
  // The keys of the jobs handed out, in order.
  const std::vector<size_t>& keys() const { return keys_; }

 private:
  size_t chunks_;
  size_t chunk_ = 0;
  Maker make_;
  std::deque<std::pair<size_t, PoolJob>> buffered_;
  std::vector<size_t> keys_;
};

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

// Labels every job `jobs` hands out, calling done(key, record) in order.
void label_all(RolloutValues& values, const PoolLabeler::Params& params, ChunkedJobs& jobs,
               const std::function<void(size_t, const PoolPosition&)>& done) {
  PoolLabeler labeler(values, params);
  size_t emitted = 0;
  labeler.run([&] { return jobs.next(); },
              [&](PoolPosition&& p) { done(jobs.keys()[emitted++], p); });
}

BobReply reply_from(const std::vector<Move>& moves, const PoolPosition& labeled) {
  const std::vector<PoolLabel>& labels = labeled.labels;
  const auto best = std::max_element(
    labels.begin(), labels.end(), [](const PoolLabel& a, const PoolLabel& b) { return a.q < b.q; });
  const size_t k = size_t(best - labels.begin());
  return {.move = moves[k], .equity_rank = int(k), .candidates = int(moves.size()), .q = best->q};
}

// One of Alice's candidates' subtrees: the jobs of the racks where Bob has a
// choice to sim, keyed by rack index, and the replies of the others in place.
struct CandidateSubtrees {
  std::vector<std::pair<size_t, PoolJob>> jobs;
  std::vector<std::vector<Move>> moves;  // Bob's candidates, by rack
};

CandidateSubtrees candidate_subtrees(const AliceTurn& turn, size_t c, const BeliefModel& model,
                                     const SubtreeParams& params, const Dictionary& dict,
                                     uint64_t seed, std::vector<BobReply>& replies) {
  const auto beliefs = bob_beliefs(turn, turn.candidates[c], model, dict);
  const size_t racks = turn.bob_racks.size();
  CandidateSubtrees out{.jobs = {}, .moves = std::vector<std::vector<Move>>(racks)};
  std::vector<std::optional<PoolJob>> jobs(racks);
  util::parallel_for(racks, [&](size_t i) {
    const SimPosition pos = bob_position(turn.applied[c], turn.bob_racks[i]);
    out.moves[i] = bob_candidates(pos, params, dict);
    if (out.moves[i].size() <= 1) {
      replies[i] = {.move = out.moves[i].empty() ? Move::pass() : out.moves[i].front()};
      return;
    }
    jobs[i] = position_job(pos, out.moves[i], rack_seed(seed, kSubtreeDomain, i), beliefs[i].get(),
                           dict, params.alice_racks);
  });
  for (size_t i = 0; i < racks; ++i)
    if (jobs[i]) out.jobs.emplace_back(i, std::move(*jobs[i]));
  return out;
}

// Alice's sims of (c, rack i) after Bob's reply: Bob has just moved, and each
// sim seats her full rack, so the sim draws only Bob's replenishment.
PoolJob after_reply_job(const AliceTurn& turn, size_t c, size_t i, const Move& reply, int sims,
                        const Dictionary& dict, uint64_t seed) {
  const RolloutStart& applied = turn.applied[c];
  const uint64_t rack_draws = rack_seed(seed, kAfterReplyDomain, i);
  PoolJob job = position_job(bob_position(applied, turn.bob_racks[i]), {reply}, rack_draws, nullptr,
                             dict, sims);
  for (int k = 0; k < sims; ++k) {
    job.group.opp_leaves.push_back(alice_rack_after_reply(
      applied, turn.bob_racks[i], util::splitmix64(rack_draws ^ uint64_t(k))));
  }
  return job;
}

}  // namespace

Rack alice_rack_after_reply(const RolloutStart& applied, const Rack& bob_rack, uint64_t seed) {
  TileCounts bag = applied.board.unseen_tiles(bob_rack);
  const bool held = bag.remove(applied.leave.counts()) && bag.remove(applied.returned.counts());
  DEBUG_ASSERT(held, "Alice's leave and exchanged tiles are unseen to Bob");
  Bag draws(seed, bag);
  Rack rack = applied.leave;
  while (rack.size() < RACK_SIZE && draws.size() > 0) rack.add(draws.draw());
  return rack;
}

size_t best_candidate(const std::vector<double>& values) {
  size_t best = values.size();
  for (size_t c = 0; c < values.size(); ++c)
    if (!std::isnan(values[c]) && (best == values.size() || values[c] > values[best])) best = c;
  return best;
}

std::vector<Rack> bob_racks(const RolloutGroup& group, const RolloutStart& start, int count) {
  RolloutBatch batch{.groups = {group}, .starts = {start}, .tasks = {}};
  batch.starts[0].group = 0;
  std::vector<Rack> out;
  out.reserve(size_t(count));
  for (int i = 0; i < count; ++i)
    out.push_back(rollout_racks(batch, {.position = 0, .rollout = uint32_t(i)}).opp);
  return out;
}

std::vector<std::vector<BobReply>> bob_replies(const AliceTurn& turn,
                                               const std::vector<size_t>& which,
                                               const BeliefModel& model, RolloutValues& values,
                                               const SubtreeParams& params, const Dictionary& dict,
                                               uint64_t seed) {
  std::vector<std::vector<BobReply>> replies(turn.candidates.size());
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
  label_all(values, {.rollouts = params.alice_racks, .race = params.race}, jobs,
            [&](size_t key, const PoolPosition& p) {
              const size_t c = key / turn.bob_racks.size();
              const size_t i = key % turn.bob_racks.size();
              replies[c][i] = reply_from(moves[c][i], p);
            });
  return replies;
}

std::vector<double> values_after_replies(const AliceTurn& turn, const std::vector<size_t>& which,
                                         const std::vector<std::vector<BobReply>>& replies,
                                         RolloutValues& values, int sims, const Dictionary& dict,
                                         uint64_t seed) {
  const size_t racks = turn.bob_racks.size();
  std::vector<double> out(turn.candidates.size(), 0.0);
  ChunkedJobs jobs(which.size(), [&](size_t chunk) {
    const size_t c = which[chunk];
    std::vector<std::pair<size_t, PoolJob>> made(racks);
    util::parallel_for(racks, [&](size_t i) {
      made[i] = {c * racks + i, after_reply_job(turn, c, i, replies[c][i].move, sims, dict, seed)};
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
