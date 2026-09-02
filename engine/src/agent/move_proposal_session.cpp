#include "agent/move_proposal_session.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace scribblez {
namespace agent {

namespace {

// `rows` of raw logits, softmax'd in place across each row's `width` floats.
void softmax_rows_in_place(float* data, int rows, int width) {
  for (int r = 0; r < rows; ++r) {
    float* row = data + size_t(r) * width;
    float m = row[0];
    for (int i = 1; i < width; ++i) m = std::max(m, row[i]);
    double sum = 0.0;
    for (int i = 0; i < width; ++i) {
      row[i] = std::exp(row[i] - m);
      sum += row[i];
    }
    for (int i = 0; i < width; ++i) row[i] = float(row[i] / sum);
  }
}

}  // namespace

MoveProposalSession::MoveProposalSession(std::shared_ptr<MoveProposalNets> nets)
    : nets_(std::move(nets)) {}

const MoveProposalPredictions& MoveProposalSession::encode(
  const float* board_row, const move_set::MoveFeatureArrays& moves) {
  nets_->run_cache(board_row, moves, &cache_);
  // The evidence-free predictions, decoded from the retained raw outputs. The
  // cache graph emits no gain head.
  plain_.num_moves = cache_.num_moves;
  plain_.wld = cache_.wld;
  plain_.score_diff = cache_.score_diff;
  plain_.gain.clear();
  softmax_rows_in_place(plain_.wld.data(), plain_.num_moves, nn::WldOutput::kRowElems);
  return plain_;
}

const MoveProposalPredictions& MoveProposalSession::condition(const EvidenceSet& evidence) {
  nets_->run_step(cache_, evidence, &conditioned_);
  softmax_rows_in_place(conditioned_.wld.data(), conditioned_.num_moves, nn::WldOutput::kRowElems);
  return conditioned_;
}

}  // namespace agent
}  // namespace scribblez
