#pragma once

// The sequential evidence loop (docs/roadmap.md "The destination", item 6): the
// decision procedure of docs/sim_residual_feedback.md at B = 1, R = K. One
// candidate is simmed at a time; after each sim the move proposal model is
// conditioned on every sim so far and re-scores every candidate, and the next
// sim is whichever unsimmed candidate the pick policy chooses off that
// conditioned pass -- until the sim budget is spent, no candidate is left, or
// the policy declines to continue.
//
//   sim the anchor                                 (no network: the highest-
//                                                   raw-score candidate)
//   while sims < max_sims and unsimmed remain:
//     conditioned = service.condition(evidence)    (every candidate re-scored)
//     pick = policy.pick(conditioned, simmed)      (nullopt = stop early)
//     sim the pick; append to the evidence
//
// Common random numbers hold across the loop: the simmer draws every
// candidate's rollout i from the same seed, so the observations pair exactly
// -- for terminal rollouts bit for bit with one batched SimRunner::run over the
// same candidates. The loop is NN-free and sim-free by construction (it drives
// a MoveProposalService and a CandidateSimmer it is handed), so its logic is
// unit-tested with scripted stubs, and its pick rule is a policy so the
// playing agent (argmax gain with an early-stopping threshold) and a future
// conditioned trajectory generator (a tempered draw) share the loop.

#include "agent/move_proposal_service.h"
#include "game/move.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace scribblez {
namespace agent {

// Sims one candidate of the turn's position, under common random numbers
// across calls.
class CandidateSimmer {
 public:
  virtual ~CandidateSimmer() = default;
  virtual SimObservation sim(const Move& candidate) = 0;
};

// The production simmer: one one-candidate SimRunner::run per sim, every call
// with the SAME base seed -- rollout i of every candidate is seeded base_seed +
// i, so the sims of a turn pair as if batched. `runner` and `pos` must outlive
// the simmer.
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

// Which unsimmed candidate to sim next, off the conditioned pass. `simmed[i]`
// marks the candidates already in the evidence set; at least one is unsimmed
// when pick() is called. Nullopt stops the loop.
class PickPolicy {
 public:
  virtual ~PickPolicy() = default;
  virtual std::optional<int> pick(const MoveProposalPredictions& conditioned,
                                  const std::vector<char>& simmed) = 0;
};

// The playing agent's rule: the unsimmed candidate with the highest proves-best
// gain (ties to the lowest scored index, static-equity order), unless even that
// gain is below `gain_threshold` -- then stop, the remaining budget being worth
// more than the candidate it would buy. A threshold of 0 never stops early
// (the gain is >= 0 by construction).
class ArgmaxGainPolicy : public PickPolicy {
 public:
  explicit ArgmaxGainPolicy(float gain_threshold) : gain_threshold_(gain_threshold) {}

  std::optional<int> pick(const MoveProposalPredictions& conditioned,
                          const std::vector<char>& simmed) override;

 private:
  float gain_threshold_;
};

// Run the loop over `candidates` -- the turn's legal moves in descending
// static-equity order, the set `service` has just been encode()d over, so
// scored index == candidate index -- for at most `max_sims` sims (>= 1).
// Returns the evidence set in sim order, the anchor first; the final pick is
// the caller's (best simmed candidate by simulation value).
EvidenceSet run_evidence_loop(const std::vector<Move>& candidates, MoveProposalService& service,
                              CandidateSimmer& simmer, PickPolicy& policy, int max_sims);

}  // namespace agent
}  // namespace scribblez
