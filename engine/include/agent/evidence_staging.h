#pragma once

// Stages sim observations as the move proposal model's evidence inputs
// (docs/roadmap.md item 3). The model runs as two ONNX graphs: a `cache` graph,
// run once per turn, computes the trunk, the per-move encodings, and the
// evidence-free predictions; a `step` graph, run once per evidence-loop
// iteration, reads those cached tensors plus the evidence staged here.
//
// Each simmed candidate becomes one evidence token of three parts:
//   - its move encoding, from the cache graph's per-move output;
//   - its sim observation: the four rollout footprint histograms and the
//     outcome moments;
//   - the model's evidence-free prediction for it: footprint planes and value.
// Giving the fusion stage observation and prediction side by side lets it
// learn the residual k*(obs - prior) rather than a correction from the
// observation alone (docs/plans/sim_residual_feedback.md).
//
// This mirrors build_evidence_inputs in py/scribblez/move_set_eval/evidence.py:
// the same normalization into the same padded (max_evidence, ...) layout, and
// the layout constants below match EVIDENCE_PLANE_NAMES / EVIDENCE_SCALAR_NAMES
// in evidence_fusion.py. A change on either side must be made on both.
// EvidenceStaging.MatchesHandComputedNormalization checks the numbers directly;
// test_proposal_inference_parity checks the whole path against Python.

#include "game/board.h"
#include "game/move.h"
#include "sim/sim_runner.h"
#include "training/footprint.h"

#include <cstdint>
#include <span>

namespace scribblez {
namespace evidence {

// Per-token spatial channels, in EVIDENCE_PLANE_NAMES order:
//   - observed: 4 heads x kSlotsPerCell, rollout footprint frequencies;
//   - predicted: 4 heads x kSlotsPerCell, the model's footprint probabilities;
//   - the candidate's own footprint: kSlotsPerCell, one-hot.
// Within a block, anchored footprint class (cell, slot) lands on channel
// (head * kSlotsPerCell + slot) at that cell. The two catch-all classes (pass,
// not-win) are dropped and nothing is renormalized.
inline constexpr int kNumPlacementHeads = 4;  // opp/self next, opp/self win
inline constexpr int kNumObservedPlanes = kNumPlacementHeads * kSlotsPerCell;
inline constexpr int kNumPredictedPlanes = kNumPlacementHeads * kSlotsPerCell;
inline constexpr int kNumEvidencePlanes = kNumObservedPlanes + kNumPredictedPlanes + kSlotsPerCell;
inline constexpr int kEvidencePlaneCells = BOARD_SIZE * BOARD_SIZE;

// Per-token scalars, in EVIDENCE_SCALAR_NAMES order:
//   - observed (6): win, draw, loss frequencies; score-delta mean and std;
//     log1p(rollouts);
//   - predicted (5): the model's win, draw, loss probabilities; its
//     score-diff mean and std.
// Score moments on both sides share one scale (evidence_staging.cpp).
inline constexpr int kNumObservedScalars = 6;
inline constexpr int kNumPredictedScalars = 5;
inline constexpr int kNumEvidenceScalars = kNumObservedScalars + kNumPredictedScalars;

// The cache graph's raw per-candidate outputs, one row per scored candidate:
// the evidence-free half of every token. The WLD logits are softmaxed here;
// the planes arrive already decoded by the graph.
struct CachePredictions {
  const float* move_enc;    // (num_scored, channels), row-major
  const float* wld_logits;  // (num_scored, 3)
  const float* score_diff;  // (num_scored, 2) = [mean, std]
  // Footprint probabilities, already in the predicted-block channel layout.
  const float* plane_probs;  // (num_scored, kNumPredictedPlanes, kEvidencePlaneCells)
  int channels;
};

// One position's padded step-graph evidence inputs.
struct EvidenceStagingOutputs {
  float* move_enc;     // (max_evidence, channels)
  float* obs_planes;   // (max_evidence, kNumEvidencePlanes, kEvidencePlaneCells)
  float* obs_scalars;  // (max_evidence, kNumEvidenceScalars)
  std::uint8_t* mask;  // (max_evidence,)
};

// Fill `out` from the simmed candidates. `moves`, `observations`, and
// `scored_indices` are parallel, in evidence order; `scored_indices[j]` locates
// candidate j in `predictions`. Rows past the evidence are zeroed, and
// `out.mask` marks the real ones. Throws if the spans differ in length or hold
// more than `max_evidence` entries.
void stage_evidence(std::span<const Move> moves, std::span<const SimObservation> observations,
                    std::span<const int> scored_indices, const CachePredictions& predictions,
                    int max_evidence, const EvidenceStagingOutputs& out);

}  // namespace evidence
}  // namespace scribblez
