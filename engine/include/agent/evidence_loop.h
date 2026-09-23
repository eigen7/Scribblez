#pragma once

// The sequential evidence loop (docs/roadmap.md item 6): the fully sequential
// decision procedure of docs/plans/sim_residual_feedback.md, one sim per
// round. After each sim the move proposal model, conditioned on every sim so
// far, re-scores every candidate, and a pick policy chooses the next sim from
// that pass:
//
//   sim the anchor                               (the highest-raw-score
//                                                 candidate; no network)
//   while sims < max_sims and unsimmed candidates remain:
//     conditioned = service.condition(evidence)  (every candidate re-scored)
//     pick = policy.pick(conditioned, simmed)    (nullopt = stop early)
//     sim the pick; append it to the evidence
//
// Every sim of a turn uses the same base seed, so rollout i of each candidate
// sees the same draws and the observations pair exactly; for terminal
// rollouts they are bit-identical to one batched SimRunner::run.
//
// The loop takes its model and simmer as interfaces, so it is unit-tested with
// scripted stubs. The pick rule is a policy so that the playing agent (argmax
// gain, with early stopping) and a future conditioned trajectory generator (a
// tempered draw) can share the loop.

#include "agent/move_proposal_service.h"
#include "game/move.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace scribblez {
namespace agent {

// Sims one candidate of the turn's position. Successive calls must use common
// random numbers.
class CandidateSimmer {
 public:
  virtual ~CandidateSimmer() = default;
  virtual SimObservation sim(const Move& candidate) = 0;
};

// The production simmer: one single-candidate SimRunner::run per call, every
// call with the same base seed, so the sims of a turn pair as if batched.
// `runner` and `pos` must outlive the simmer.
class SimRunnerCandidateSimmer : public CandidateSimmer {
 public:
  SimRunnerCandidateSimmer(const SimRunner& runner, const SimPosition& pos, uint64_t base_seed)
      : runner_(runner), pos_(pos), base_seed_(base_seed) {}

  SimObservation sim(const Move& candidate) override;

 private:
  const SimRunner& runner_;
  const SimPosition& pos_;
  uint64_t base_seed_;
};

// Chooses the next candidate to sim from the conditioned pass. `simmed[i]`
// marks the candidates already in the evidence set; at least one is unsimmed
// when pick() is called. Nullopt stops the loop.
class PickPolicy {
 public:
  virtual ~PickPolicy() = default;
  virtual std::optional<int> pick(const MoveProposalPredictions& conditioned,
                                  const std::vector<char>& simmed) = 0;
};

// The playing agent's rule: the unsimmed candidate with the highest
// proves-best gain, ties going to the lowest index (the static-equity
// favourite). Stops instead when even that gain is below `gain_threshold`.
// A threshold of 0 never stops, since the gain is >= 0 by construction.
class ArgmaxGainPolicy : public PickPolicy {
 public:
  explicit ArgmaxGainPolicy(float gain_threshold) : gain_threshold_(gain_threshold) {}

  std::optional<int> pick(const MoveProposalPredictions& conditioned,
                          const std::vector<char>& simmed) override;

 private:
  float gain_threshold_;
};

// Run the loop for at most `max_sims` (>= 1) sims. `candidates` are the
// turn's legal moves in descending static-equity order, exactly the set
// `service` was last encode()d over, so a candidate's index is its scored
// index. Returns the evidence set in sim order, anchor first; choosing the
// move to play from it is the caller's job.
EvidenceSet run_evidence_loop(const std::vector<Move>& candidates, MoveProposalService& service,
                              CandidateSimmer& simmer, PickPolicy& policy, int max_sims);

}  // namespace agent
}  // namespace scribblez
