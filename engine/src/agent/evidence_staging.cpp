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

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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

// The four observed rollout-frequency planes (count / rollouts) followed by the
// four evidence-free predicted planes (sigmoid of the cache's plane logits) and
// the candidate's footprint -- kNumEvidencePlanes planes of kCells each.
void stage_planes(const SimObservation& obs, const Move& move, const float* plane_logits,
                  float* out) {
  const float inv_n = 1.0f / float(std::max<std::uint32_t>(obs.n, 1));
  for (int cell = 0; cell < kCells; ++cell) {
    out[0 * kCells + cell] = float(obs.opp_next_count[cell]) * inv_n;
    out[1 * kCells + cell] = float(obs.self_next_count[cell]) * inv_n;
    out[2 * kCells + cell] = obs.opp_win_count[cell] * inv_n;
    out[3 * kCells + cell] = obs.self_win_count[cell] * inv_n;
  }
  for (int p = 0; p < kNumPredictedPlanes; ++p) {
    float* dst = out + (kNumObservedPlanes + p) * kCells;
    const float* src = plane_logits + p * kCells;
    for (int cell = 0; cell < kCells; ++cell) dst[cell] = sigmoidf(src[cell]);
  }
  float* footprint = out + (kNumEvidencePlanes - 1) * kCells;
  visit_placed_squares(move, [&](int r, int c) { footprint[r * BOARD_SIZE + c] = 1.0f; });
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
                 predictions.plane_logits + size_t(idx) * kNumPredictedPlanes * kCells,
                 out.obs_planes + size_t(j) * kNumEvidencePlanes * kCells);
    stage_scalars(observations[j], predictions.wld_logits + size_t(idx) * 3,
                  predictions.score_diff + size_t(idx) * 2,
                  out.obs_scalars + size_t(j) * kNumEvidenceScalars);
    out.mask[j] = 1;
  }
}

}  // namespace evidence
}  // namespace scribblez
