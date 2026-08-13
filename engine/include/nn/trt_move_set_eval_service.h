#pragma once

#include "nn/move_set_eval_service.h"
#include "nn/move_set_net.h"

#include <memory>

// The TensorRT-backed MoveSetEvalService: the production implementation of the
// interface, kept out of move_set_eval_service.h so an agent or a stub-driven
// test depends on the interface alone -- as nn_evaluation_service.h is to
// eval_service.h for the position model.

namespace scribblez {
namespace nn {

class TrtMoveSetEvalService : public MoveSetEvalService {
 public:
  explicit TrtMoveSetEvalService(const MoveSetNetParams& params);

  // Call once before evaluate().
  void load();

  // Valid after load().
  bool contingent_features() const override { return net_.contingent_features(); }
  bool opp_leave_input() const override { return net_.opp_leave_input(); }
  int spatial_planes() const override { return net_.spatial_planes(); }
  int scalar_floats() const override { return net_.scalar_floats(); }

  // Blocks until inference completes. A candidate set larger than the engine's
  // max_moves is split into chunks, which changes no result: given the board,
  // the model scores each candidate independently.
  void evaluate(const float* board_row, const move_set::MoveFeatureArrays& moves,
                Eval* out) override;

 private:
  // Score `chunk` moves starting at `start` within `moves`, the board already
  // staged.
  void evaluate_chunk(const move_set::MoveFeatureArrays& moves, int start, int chunk, Eval* out);

  MoveSetNet net_;
};

// Construct and load() the service for `params` -- the one call an agent
// factory needs to stand up production move set inference.
std::unique_ptr<MoveSetEvalService> make_loaded_move_set_service(const MoveSetNetParams& params);

}  // namespace nn
}  // namespace scribblez
