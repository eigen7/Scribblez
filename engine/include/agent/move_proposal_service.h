#pragma once

// The move proposal model as the evidence loop sees it (docs/roadmap.md items
// 3 and 6), behind an interface with no CUDA/TensorRT dependency: production
// uses MoveProposalSession (move_proposal_session.h), tests use scripted stubs.
// This is the same seam nn::EvalService provides for the position and move-set
// model families.
//
// The predictions carry no placement planes, since no loop consumer reads
// them. The planes the evidence tokens need are gathered from the session's
// retained cache when the evidence is staged.

#include "game/move.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"

#include <vector>

namespace scribblez {
namespace agent {

// Decoded predictions for the M encoded candidates, in encoding order.
struct MoveProposalPredictions {
  int num_moves = 0;
  std::vector<float> wld;         // (M, 3) probabilities [win, draw, loss]
  std::vector<float> score_diff;  // (M, 2) [mean, std] in score points
  // (M,) the proves-best expected gain (>= 0), in win-probability units.
  // Empty after encode(), whose graph has no gain head; condition() over an
  // empty evidence set gives the unconditioned gain.
  std::vector<float> gain;
};

// The simmed candidates, in sim order. `scored_indices[j]` is candidate j's
// index in the encoded candidate set, which locates its cached encoding and
// predicted planes.
struct EvidenceSet {
  std::vector<Move> moves;
  std::vector<SimObservation> observations;
  std::vector<int> scored_indices;

  int size() const { return int(moves.size()); }
  void clear() { *this = EvidenceSet{}; }
  void add(const Move& move, const SimObservation& observation, int scored_index) {
    moves.push_back(move);
    observations.push_back(observation);
    scored_indices.push_back(scored_index);
  }
};

class MoveProposalService : public nn::ServedModelInputs {
 public:
  // Once per turn: the evidence-free pass over one position's candidate set.
  // `board_row` is [spatial | scalar] floats as GameStateEncoder::encode_input
  // writes them. Retains the position for condition() until the next encode().
  virtual const MoveProposalPredictions& encode(const float* board_row,
                                                const move_set::MoveFeatureArrays& moves) = 0;

  // Once per loop iteration: re-score the encoded candidates conditioned on
  // `evidence`, including the gain the loop picks by. An empty set reproduces
  // encode()'s predictions within tolerance. Throws on a set wider than
  // nn::kMaxEvidence.
  virtual const MoveProposalPredictions& condition(const EvidenceSet& evidence) = 0;
};

}  // namespace agent
}  // namespace scribblez
