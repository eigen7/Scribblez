#include "agent/evidence_staging.h"

#include "util/exception.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace scribblez {
namespace evidence {
namespace {

constexpr int kCells = kEvidencePlaneCells;
// Delta moments are carried in score points scaled to ~unit range, and the
// rollout count through a log1p compressed by the same factor, matching
// evidence.py so the trained fusion stage sees the inputs it was fitted on.
constexpr double kScorePointScale = 100.0;
constexpr double kRolloutLogScale = 8.0;

// Softmax of one candidate's three WLD logits into [p_win, p_draw, p_loss],
// numerically stable (the same decode the eval service applies to this head).
void softmax3(const float* logits, float* out) {
  const float m = std::max({logits[0], logits[1], logits[2]});
  double sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    out[i] = std::exp(logits[i] - m);
    sum += out[i];
  }
  for (int i = 0; i < 3; ++i) out[i] = float(out[i] / sum);
}

// One head's dense footprint histogram scattered into its kSlotsPerCell
// channels: anchored class (cell, slot) -> channel `slot` at `cell`, scaled by
// inv_n. The anchored classes tile the head's channels exactly; the two
// catch-all classes are dropped.
template <typename T>
void scatter_histogram(const T* hist, float inv_n, float* out) {
  for (int cls = 0; cls < kAnchoredFootprints; ++cls)
    out[(cls % kSlotsPerCell) * kCells + cls / kSlotsPerCell] = float(hist[cls]) * inv_n;
}

// The observed rollout-frequency channels (histogram counts / rollouts)
// followed by the evidence-free predicted channels (the model's footprint
// probabilities, copied through) and the candidate's own footprint one-hot --
// kNumEvidencePlanes planes of kCells each. `out` arrives zeroed.
void stage_planes(const SimObservation& obs, const Move& move, const float* plane_probs,
                  float* out) {
  const float inv_n = 1.0f / float(std::max<std::uint32_t>(obs.n, 1));
  scatter_histogram(obs.opp_next_count.data(), inv_n, out + 0 * kSlotsPerCell * kCells);
  scatter_histogram(obs.self_next_count.data(), inv_n, out + 1 * kSlotsPerCell * kCells);
  scatter_histogram(obs.opp_win_count.data(), inv_n, out + 2 * kSlotsPerCell * kCells);
  scatter_histogram(obs.self_win_count.data(), inv_n, out + 3 * kSlotsPerCell * kCells);
  // The model's predicted planes arrive already softmaxed and already in the
  // evidence-channel layout (catch-all dropped in the graph): a plain copy.
  std::memcpy(out + kNumObservedPlanes * kCells, plane_probs,
              sizeof(float) * kNumPredictedPlanes * kCells);
  // The candidate's own footprint, one-hot at (slot channel, anchor cell);
  // a PASS/EXCHANGE maps to a catch-all class and leaves the block zero.
  const int cls = footprint_class(move, /*flip=*/false);
  if (cls < kAnchoredFootprints) {
    float* footprint = out + (kNumObservedPlanes + kNumPredictedPlanes) * kCells;
    footprint[(cls % kSlotsPerCell) * kCells + cls / kSlotsPerCell] = 1.0f;
  }
}

// The six observed scalars (WDL frequencies, delta mean/std in scaled score
// points, log1p rollouts) followed by the five predicted scalars (the cache's
// WLD probabilities and its score-diff moments, scaled like the observed ones).
void stage_scalars(const SimObservation& obs, const float* wld_logits, const float* score_diff,
                   float* out) {
  const double n = std::max<double>(obs.n, 1.0);
  const double delta_mean = obs.delta_sum / n;
  const double delta_var = std::max(obs.delta_sq_sum / n - delta_mean * delta_mean, 0.0);
  out[0] = float(obs.wins / n);
  out[1] = float(obs.draws / n);
  out[2] = float(obs.losses / n);
  out[3] = float(delta_mean / kScorePointScale);
  out[4] = float(std::sqrt(delta_var) / kScorePointScale);
  out[5] = float(std::log1p(double(obs.n)) / kRolloutLogScale);
  softmax3(wld_logits, out + kNumObservedScalars);
  out[kNumObservedScalars + 3] = float(score_diff[0] / kScorePointScale);
  out[kNumObservedScalars + 4] = float(score_diff[1] / kScorePointScale);
}

}  // namespace

void stage_evidence(std::span<const Move> moves, std::span<const SimObservation> observations,
                    std::span<const int> scored_indices, const CachePredictions& predictions,
                    int max_evidence, const EvidenceStagingOutputs& out) {
  const int num_evidence = int(moves.size());
  if (int(observations.size()) != num_evidence || int(scored_indices.size()) != num_evidence)
    throw util::Exception("evidence moves, observations, and indices must be parallel arrays");
  if (num_evidence > max_evidence)
    throw util::Exception("evidence set of {} does not fit the step graph's padded width {}",
                          num_evidence, max_evidence);
  const int c = predictions.channels;

  // Padded rows carry nothing: zero everything, mark only the real rows, then
  // fill them (the fusion stage's mask gates the padding out regardless, but a
  // deterministic zero keeps the inputs reproducible).
  std::memset(out.move_enc, 0, sizeof(float) * size_t(max_evidence) * c);
  std::memset(out.obs_planes, 0,
              sizeof(float) * size_t(max_evidence) * kNumEvidencePlanes * kCells);
  std::memset(out.obs_scalars, 0, sizeof(float) * size_t(max_evidence) * kNumEvidenceScalars);
  std::memset(out.mask, 0, sizeof(std::uint8_t) * max_evidence);

  for (int j = 0; j < num_evidence; ++j) {
    const int idx = scored_indices[j];
    std::memcpy(out.move_enc + size_t(j) * c, predictions.move_enc + size_t(idx) * c,
                sizeof(float) * c);
    stage_planes(observations[j], moves[j],
                 predictions.plane_probs + size_t(idx) * kNumPredictedPlanes * kCells,
                 out.obs_planes + size_t(j) * kNumEvidencePlanes * kCells);
    stage_scalars(observations[j], predictions.wld_logits + size_t(idx) * 3,
                  predictions.score_diff + size_t(idx) * 2,
                  out.obs_scalars + size_t(j) * kNumEvidenceScalars);
    out.mask[j] = 1;
  }
}

}  // namespace evidence
}  // namespace scribblez
