#pragma once

// Drives the move proposal model's two-graph evidence-path runtime (roadmap
// item 3). The model runs incrementally: once per turn a `cache` graph encodes
// the board, the M candidates, and the evidence-free predictions; then, after
// each sim, a `step` graph conditions on the growing evidence set and re-scores
// every candidate without recomputing the trunk (docs/sim_residual_feedback.md,
// docs/roadmap.md). This class owns the two NeuralNet<Spec> engines and marshals
// the handoff between them:
//
//   encode(board_row, moves)   -- runs the cache graph over one position, retains
//                                 the full-M board/g/move_enc handoff tensors and
//                                 the raw per-candidate predictions, and returns
//                                 the decoded evidence-free predictions.
//   condition(evidence set)    -- stages the evidence (agent/evidence_staging.h),
//                                 copies the cached handoff tensors into the step
//                                 graph's inputs, runs it, and returns the
//                                 evidence-conditioned predictions.
//
// The two nets are driven DIRECTLY (net.predict() + net.host<Tensor>()), not
// through TrtEvalService: that service scales every output by the chunk row
// count in a uniform decode loop, which cannot serve the cache's static
// (board/g) or raw (move_enc) handoff outputs. This class instead reads those
// off the host buffers itself and applies each head's activation (softmax on the
// WLD head; the plane head is already softmaxed to a per-cell marginal in the
// graph) here.
//
// SCOPE: this is the runtime that drives and tests the two-graph loop today, not
// the finished production surface. The sequential *playing* agent -- the loop
// that alternates sims and condition() calls and reads the proves-best gain to
// decide when to stop -- is roadmap item 6; this API is expected to be revisited
// once item 6 defines that calling pattern (e.g. a device-resident handoff that
// avoids the host round trip this class accepts for item 3). For now it holds
// one position at a time: encode() replaces the retained cache, condition()
// reads it.

#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/trt_util.h"
#include "training/move_set_encoder.h"

#include <span>
#include <string>
#include <vector>

namespace scribblez {

struct SimObservation;  // sim/sim_runner.h
class Move;             // game/move.h

namespace agent {

// One candidate set's decoded predictions -- the plain pass from encode() or the
// evidence-conditioned pass from condition(). Rows are M candidates in the order
// they were encoded.
struct MoveProposalPredictions {
  int num_moves = 0;
  std::vector<float> wld;         // (M, 3) probabilities [win, draw, loss]
  std::vector<float> score_diff;  // (M, 2) [mean, std] in score points
  std::vector<float> planes;      // (M, 4, 225) per-cell probabilities
  // (M,) the proves-best expected gain (>= 0). Populated by condition() only:
  // the cache graph emits no gain head, so encode()'s plain pass leaves it
  // empty (condition() over an empty set recovers the plain gain).
  std::vector<float> gain;
};

class MoveProposalRuntime {
 public:
  struct Params {
    std::string cache_onnx_path;
    std::string step_onnx_path;
    int cuda_device_id = 0;
    // The candidate ceiling per predict(); candidate sets above it are chunked.
    int max_rows = nn::MoveProposalCacheSpec::kDefaultMaxRows;
    // FP32 for item 3: correctness and the parity contract, not FP16 speed --
    // the fusion graph's masked_fill / 4D einsum are FP16 hazards gated
    // separately (docs/fp16_safe_serving.md). Callers may override, with that
    // caveat.
    nn::Precision precision = nn::Precision::kFP32;
    bool fast_build = false;
    std::string mount_root = "/workspace/mount";
  };

  explicit MoveProposalRuntime(const Params& params);

  // Build or load both graphs' engines, then validate the pair came from one
  // model: they must share the proposal_export_id fingerprint and agree on the
  // trunk channel width C. Throws otherwise. Exactly once, before encode().
  void load();

  // The trunk channel width C, and the padded evidence width E the step graph
  // is specialized to. Valid after load().
  int channels() const { return cache_net_.channels(); }
  int max_evidence() const { return nn::kMaxEvidence; }

  // The cache graph's board-row widths, for a caller sizing the encoder row it
  // passes to encode(). Valid after load().
  int spatial_planes() const { return cache_net_.spatial_planes(); }
  int scalar_floats() const { return cache_net_.scalar_floats(); }

  // Encode one position: `board_row` is [spatial | scalar] floats as
  // GameStateEncoder::encode_input writes them; `moves` its candidate set.
  // Retains the cache internally and returns the evidence-free predictions
  // (no gain head; see MoveProposalPredictions::gain).
  const MoveProposalPredictions& encode(const float* board_row,
                                        const move_set::MoveFeatureArrays& moves);

  // Re-score the encoded candidate set conditioned on an evidence set: the
  // simmed candidates in evidence order, each locating its cache row by
  // `scored_indices[j]` (agent/evidence_staging.h). Must follow encode().
  // An empty set returns the plain predictions within tolerance (the fusion
  // hard-gate); a set larger than max_evidence() throws (from staging).
  const MoveProposalPredictions& condition(std::span<const Move> moves,
                                           std::span<const SimObservation> observations,
                                           std::span<const int> scored_indices);

 private:
  Params params_;
  nn::NeuralNet<nn::MoveProposalCacheSpec> cache_net_;
  nn::NeuralNet<nn::MoveProposalStepSpec> step_net_;

  // The current position's retained cache: the raw per-candidate outputs the
  // step graph's evidence staging gathers from (full M, held across chunk
  // boundaries -- the nets' own host buffers hold only max_rows rows), and the
  // static board/g handoff tensors. Sized by encode().
  int num_moves_ = 0;
  std::vector<float> cache_move_enc_;    // (M, C) raw
  std::vector<float> cache_wld_;         // (M, 3) raw logits
  std::vector<float> cache_score_diff_;  // (M, 2) [mean, std]
  std::vector<float> cache_planes_;      // (M, 4, 225) per-cell probs (anchor marginal)
  std::vector<float> cache_board_;       // (225, C) raw
  std::vector<float> cache_g_;           // (3C,) raw

  MoveProposalPredictions plain_;
  MoveProposalPredictions conditioned_;
};

}  // namespace agent
}  // namespace scribblez
