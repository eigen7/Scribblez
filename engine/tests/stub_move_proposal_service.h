#pragma once

// A scripted agent::MoveProposalService stub for the evidence loop's and
// UltimateBot's unit tests -- no ONNX, no TensorRT, no GPU. It declares the base
// input layout (no opponent-leave block), keeps what encode() was handed (the
// board row and the encoded candidate set, for an encoding check), records
// every evidence set condition() saw (in call order, as scored indices), and
// answers each condition() call with the next scripted gain vector -- so a test
// dictates the loop's picks sim by sim and then checks the sequence the loop
// actually took.

#include "agent/move_proposal_service.h"
#include "encoding/input_encoder.h"
#include "training/move_set_encoder.h"

#include <vector>

namespace scribblez {
namespace testing {

class StubMoveProposalService : public agent::MoveProposalService {
 public:
  // scripted_gains[k] is the (M,) gain vector the k-th condition() call
  // returns; a call past the end (or a vector shorter than M) reads zeros.
  std::vector<std::vector<float>> scripted_gains;

  int encode_calls = 0;
  int condition_calls = 0;
  std::vector<float> last_board_row;
  move_set::MoveFeatureArrays last_moves;
  std::vector<std::vector<int>> seen_evidence;  // per condition() call

  bool opp_leave_input() const override { return false; }
  int spatial_planes() const override { return scribblez::spatial_planes(); }
  int scalar_floats() const override { return scribblez::scalar_floats({nullptr}); }

  const agent::MoveProposalPredictions& encode(const float* board_row,
                                               const move_set::MoveFeatureArrays& moves) override {
    ++encode_calls;
    last_board_row.assign(board_row, board_row + input_floats(InputEncodingSpec{nullptr}));
    last_moves = moves;
    fill(moves.count, /*gain_call=*/-1);
    return predictions_;
  }

  const agent::MoveProposalPredictions& condition(const agent::EvidenceSet& evidence) override {
    seen_evidence.push_back(evidence.scored_indices);
    fill(last_moves.count, condition_calls++);
    return predictions_;
  }

 private:
  // Uniform value heads (the loop reads none of them) and the scripted gain
  // for the given condition() call, or no gain at all for the plain pass.
  void fill(int num_moves, int gain_call) {
    predictions_.num_moves = num_moves;
    predictions_.wld.assign(size_t(num_moves) * 3, 1.0f / 3.0f);
    predictions_.score_diff.assign(size_t(num_moves) * 2, 0.0f);
    predictions_.gain.clear();
    if (gain_call < 0) return;
    predictions_.gain.assign(size_t(num_moves), 0.0f);
    if (gain_call < int(scripted_gains.size())) {
      const std::vector<float>& g = scripted_gains[size_t(gain_call)];
      for (size_t i = 0; i < g.size() && i < size_t(num_moves); ++i) predictions_.gain[i] = g[i];
    }
  }

  agent::MoveProposalPredictions predictions_;
};

}  // namespace testing
}  // namespace scribblez
