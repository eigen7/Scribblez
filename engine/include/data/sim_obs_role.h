#pragma once

// The evidence role of a simmed trajectory candidate (docs/roadmap.md item 4),
// stored per SimObsRecord in a .sobs file.
//
//   * anchor, on_policy: evidence-eligible. A training row's evidence set is
//     any subset of these that contains the anchor. Each one is also a labeled
//     held-out row for the sets it is not in.
//   * off_policy: labels only. Simmed for its proves-best gain but never put in
//     an evidence set, because deployed evidence holds only the anchor and the
//     proposer's picks (docs/plans/sim_residual_feedback.md).
//
// It lives in its own header so the NN-free trajectory selection code
// (training/evidence_trajectory_select.h) can use it without pulling in the
// sim dependencies of sim_observation_log.h.

#include <cstdint>

namespace scribblez {

enum class SimObsRole : uint8_t {
  kAnchor = 0,     // the highest-raw-score move
  kOnPolicy = 1,   // a proposer pick
  kOffPolicy = 2,  // drawn uniformly from the remaining legal moves
};

}  // namespace scribblez
