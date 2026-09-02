#pragma once

// Marshals sim observations into the move proposal model's evidence inputs
// (roadmap item 3). That model runs incrementally as two ONNX graphs: a `cache`
// graph, run once per turn for the trunk, per-move encodings, and evidence-free
// predictions; and a `step` graph, run per evidence-loop iteration, which reads
// those cached tensors plus the evidence set this code stages. Each simmed
// candidate becomes one evidence token carrying three things the fusion stage
// reads side by side: the candidate's move encoding (gathered from the cache
// graph's own per-move output), its raw sim observation (the four rollout
// footprint histograms and the outcome moments), and the model's own
// evidence-free prediction for that candidate (its footprint planes and its
// value). Feeding observation and prediction in together is what lets the
// fusion stage form the residual `k*(obs - prior)` rather than an
// observation-marginal correction (docs/sim_residual_feedback.md).
//
// This is the C++ port of py/scribblez/move_set_eval/evidence.py's
// build_evidence_inputs: the SAME normalization (histogram counts / rollouts,
// delta moments in score points, log1p rollouts, softmax'd WLD, the model's
// already-decoded predicted planes) into the SAME padded (max_evidence, ...)
// layout the step graph's leading-1 evidence inputs expect. The layout constants
// below mirror evidence_fusion.py's EVIDENCE_PLANE_NAMES / EVIDENCE_SCALAR_NAMES;
// a change on either side must be mirrored on the other. The runtime's
// end-to-end parity test (test_proposal_inference_parity) ties the two sides
// together; MatchesHandComputedNormalization is the direct numeric check.

#include "game/board.h"
#include "game/move.h"
#include "sim/sim_runner.h"
#include "training/footprint.h"

#include <cstdint>
#include <span>

namespace scribblez {
namespace evidence {

// Per-token spatial channels, in EVIDENCE_PLANE_NAMES order. Placement is
// footprint-categorical: each of the four heads (observed rollout-frequency
// histograms, then the model's predicted footprint distributions) contributes
// kSlotsPerCell channels -- anchored class (cell, slot) lands on channel
// (head * kSlotsPerCell + slot) at that cell -- and the candidate's own
// footprint is a one-hot in a final kSlotsPerCell-channel block. The two
// catch-all classes (pass / not-win) are dropped, and nothing is renormalized:
// the fusion conv sees the raw per-class frequencies/probabilities.
inline constexpr int kNumPlacementHeads = 4;  // opp/self next, opp/self win
inline constexpr int kNumObservedPlanes = kNumPlacementHeads * kSlotsPerCell;
inline constexpr int kNumPredictedPlanes = kNumPlacementHeads * kSlotsPerCell;
inline constexpr int kNumEvidencePlanes = kNumObservedPlanes + kNumPredictedPlanes + kSlotsPerCell;
inline constexpr int kEvidencePlaneCells = BOARD_SIZE * BOARD_SIZE;

// Per-token scalars, in EVIDENCE_SCALAR_NAMES order: six observed (win/draw/loss
// frequency, delta mean and std in score points scaled to ~unit range, log1p
// rollouts) then five predicted (the model's WLD probabilities and its
// score-diff mean/std, the moments scaled the same way as the observed ones).
inline constexpr int kNumObservedScalars = 6;
inline constexpr int kNumPredictedScalars = 5;
inline constexpr int kNumEvidenceScalars = kNumObservedScalars + kNumPredictedScalars;

// The cache graph's raw per-candidate outputs, one row per SCORED candidate --
// the evidence-free half of every token. `move_enc` rows are gathered by a
// candidate's scored index; the WLD logits are decoded here (softmax), while the
// plane values arrive already decoded -- softmaxed footprint probabilities in
// the (head*slot, cell) channel layout above, catch-all dropped in the graph --
// exactly as evidence.py handles the first pass.
struct CachePredictions {
  const float* move_enc;    // (num_scored, channels), row-major
  const float* wld_logits;  // (num_scored, 3)
  const float* score_diff;  // (num_scored, 2) = [mean, std]
  // footprint probabilities already in evidence-channel layout, not logits:
  const float* plane_probs;  // (num_scored, kNumPredictedPlanes, kEvidencePlaneCells)
  int channels;
};

// One position's padded step-graph evidence input buffers, each sized
// `max_evidence * <per-row width>` -- bundled the way CachePredictions bundles
// the inputs. `move_enc` is (max_evidence, channels), `obs_planes`
// (max_evidence, kNumEvidencePlanes, kEvidencePlaneCells), `obs_scalars`
// (max_evidence, kNumEvidenceScalars), `mask` (max_evidence,).
struct EvidenceStagingOutputs {
  float* move_enc;
  float* obs_planes;
  float* obs_scalars;
  std::uint8_t* mask;
};

// Fills `out` (one position's padded step-graph evidence inputs) from the simmed
// candidates. `moves`, `observations`, and `scored_indices` are those candidates
// in evidence order (length num_evidence <= max_evidence); `scored_indices[j]`
// locates candidate j in the cache's per-candidate outputs. Rows past
// num_evidence are zeroed and `out.mask` marks the real ones. Throws if the
// three spans differ in length, or if num_evidence exceeds max_evidence.
void stage_evidence(std::span<const Move> moves, std::span<const SimObservation> observations,
                    std::span<const int> scored_indices, const CachePredictions& predictions,
                    int max_evidence, const EvidenceStagingOutputs& out);

}  // namespace evidence
}  // namespace scribblez
