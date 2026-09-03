#pragma once

// The move proposal model as its consumers see it (roadmap items 3 and 6): the
// two calls of the evidence loop, behind an interface that carries no
// CUDA/TensorRT dependency. A loop consumer -- the sequential playing agent,
// the conditioned trajectory generator, their unit tests -- holds a
// MoveProposalService and never learns whether it is the TensorRT-backed
// MoveProposalSession (agent/move_proposal_session.h) or a scripted stub, the
// same seam nn::EvalService cuts for the position and move-set families.
//
//   encode(board_row, moves)   -- once per turn: the model's evidence-free pass
//                                 over one position's whole candidate set.
//   condition(evidence)        -- once per loop iteration: every candidate
//                                 re-scored given the simmed candidates so far,
//                                 plus the proves-best gain the loop picks by.
//
// The predictions deliberately carry no per-candidate placement planes: no
// consumer of the loop reads them. The planes the model DOES need -- the
// evidence-free predicted planes of each simmed candidate, one half of its
// evidence token -- are gathered by scored index from the session's retained
// cache when the evidence is staged, never handed out per candidate.

#include "game/move.h"
#include "nn/eval_service.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"

#include <vector>

namespace scribblez {
namespace agent {

// One candidate set's decoded predictions -- the evidence-free pass from
// encode() or the evidence-conditioned pass from condition(). Rows are the M
// candidates in the order they were encoded.
struct MoveProposalPredictions {
  int num_moves = 0;
  std::vector<float> wld;         // (M, 3) probabilities [win, draw, loss]
  std::vector<float> score_diff;  // (M, 2) [mean, std] in score points
  // (M,) the proves-best expected gain (>= 0), in win-probability units.
  // Populated by condition() only: the cache graph emits no gain head, so
  // encode()'s pass leaves it empty (condition() over an empty set recovers
  // the plain gain).
  std::vector<float> gain;
};

// The simmed candidates a conditioned pass is given, in sim order: each one's
// move, its rollout observation, and `scored_indices[j]` -- where candidate j
// sits in the encoded candidate set, which is how its cached move encoding and
// predicted planes are found. Grows by one per loop iteration.
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
  // The padded evidence width the model is specialized to: the most simmed
  // candidates one condition() call may carry.
  virtual int max_evidence() const = 0;

  // Encode one position: `board_row` is [spatial | scalar] floats as
  // GameStateEncoder::encode_input writes them; `moves` its candidate set.
  // Returns the evidence-free predictions and retains the position for
  // condition() -- a second encode() replaces it.
  virtual const MoveProposalPredictions& encode(const float* board_row,
                                                const move_set::MoveFeatureArrays& moves) = 0;

  // Re-score the encoded candidate set conditioned on `evidence`. Must follow
  // encode(). An empty set returns the plain predictions (plus the plain gain)
  // within tolerance; a set larger than max_evidence() throws.
  virtual const MoveProposalPredictions& condition(const EvidenceSet& evidence) = 0;
};

}  // namespace agent
}  // namespace scribblez
