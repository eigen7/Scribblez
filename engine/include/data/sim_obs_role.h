#pragma once

// The evidence-eligibility role of a simmed trajectory candidate
// (docs/roadmap.md item 4), stored per SimObsRecord in a .sobs file. It is a
// property of the record, not of its position, so a reader recovers
// evidence-eligibility by role rather than by storage order:
//
//   * anchor / on_policy -- evidence-eligible. A training row's evidence set is
//     any subset of these that contains the anchor; every one of them is also a
//     labeled held-out row against the sets it is not in.
//   * off_policy -- labels-only. Simmed for its proves-best gain, but never
//     placed in an evidence set, because deployed evidence holds only the anchor
//     and the proposer's picks (docs/sim_residual_feedback.md).
//
// The one-byte enum lives in its own header so the NN-free trajectory selection
// core (training/evidence_trajectory_select.h) can name it without pulling in
// sim_observation_log.h's sim/rollout dependencies.

#include <cstdint>

namespace scribblez {

enum class SimObsRole : uint8_t {
  kAnchor = 0,     // the greedy highest-raw-score move; every evidence set holds it
  kOnPolicy = 1,   // a proposer pick; evidence-eligible
  kOffPolicy = 2,  // a uniform off-policy draw; labels-only, never in an evidence set
};

}  // namespace scribblez
