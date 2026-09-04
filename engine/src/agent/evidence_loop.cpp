#include "agent/evidence_loop.h"

#include "training/evidence_trajectory_select.h"
#include "util/assert.h"
#include "util/exception.h"

#include <cmath>

namespace scribblez {
namespace agent {

SimObservation SimRunnerCandidateSimmer::sim(const Move& candidate) {
  return runner_.run(pos_, {candidate}, base_seed_).front();
}

std::optional<int> ArgmaxGainPolicy::pick(const MoveProposalPredictions& conditioned,
                                          const std::vector<char>& simmed) {
  int best = -1;
  for (int i = 0; i < conditioned.num_moves; ++i) {
    if (simmed[size_t(i)]) continue;
    // A non-finite gain compares false against everything: a NaN at the
    // lowest unsimmed index would win every pick and slip past the threshold.
    // As for the leaf readout (sim_runner.cpp), a broken model output is a
    // hard error, not a silent decision.
    if (!std::isfinite(conditioned.gain[size_t(i)])) {
      throw util::Exception(
        "evidence loop: the move proposal model returned a non-finite gain (off-distribution "
        "input, or a broken model)");
    }
    // Strict: the first (lowest-index, equity-preferred) maximum wins a tie.
    if (best < 0 || conditioned.gain[size_t(i)] > conditioned.gain[size_t(best)]) best = i;
  }
  RELEASE_ASSERT(best >= 0);
  if (conditioned.gain[size_t(best)] < gain_threshold_) return std::nullopt;
  return best;
}

EvidenceSet run_evidence_loop(const std::vector<Move>& candidates, MoveProposalService& service,
                              CandidateSimmer& simmer, PickPolicy& policy, int max_sims) {
  const int n = int(candidates.size());
  RELEASE_ASSERT(n >= 1 && max_sims >= 1);
  EvidenceSet evidence;
  std::vector<char> simmed(size_t(n), 0);

  const int anchor = int(evidence::anchor_index(candidates));
  evidence.add(candidates[size_t(anchor)], simmer.sim(candidates[size_t(anchor)]), anchor);
  simmed[size_t(anchor)] = 1;

  while (evidence.size() < max_sims && evidence.size() < n) {
    const std::optional<int> pick = policy.pick(service.condition(evidence), simmed);
    if (!pick) break;
    const int i = *pick;
    RELEASE_ASSERT(i >= 0 && i < n && !simmed[size_t(i)]);
    evidence.add(candidates[size_t(i)], simmer.sim(candidates[size_t(i)]), i);
    simmed[size_t(i)] = 1;
  }
  return evidence;
}

}  // namespace agent
}  // namespace scribblez
