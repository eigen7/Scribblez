#pragma once

#include "nn/eval_service.h"
#include "nn/move_set_net.h"
#include "training/move_set_encoder.h"

#include <memory>

// Synchronous front-end over a MoveSetNet: one position's encoder row plus its
// encoded candidate set in, one nn::Eval per candidate out.
//
// The Eval is deliberately the same struct the position evaluation service
// produces, so an agent's EvalObjective ranks candidates identically whichever
// model produced them -- the two differ in how a value is obtained (a forward
// pass per resulting position, or one pass over the whole candidate set), not
// in what the value means.

namespace scribblez {
namespace nn {

// Abstract so agents and their GPU-free unit tests depend on this interface
// rather than on TensorRT, injecting either the TensorRT-backed service or a
// scripted stub -- as EvalService already does for the position model.
class MoveSetEvalService {
 public:
  virtual ~MoveSetEvalService() = default;

  // The served model's input-encoding arm and the board-row widths it consumes.
  // An agent builds its InputEncodingSpec from the arm and validates the widths
  // against it through the layout registry.
  virtual bool contingent_features() const = 0;
  virtual bool opp_leave_input() const = 0;
  virtual int spatial_planes() const = 0;
  virtual int scalar_floats() const = 0;

  // `board_row` is the pre-move position, laid out as
  // GameStateEncoder::encode_input() writes it; `moves` are that position's
  // candidates. Writes moves.count Evals to `out`.
  virtual void evaluate(const float* board_row, const move_set::MoveFeatureArrays& moves,
                        Eval* out) = 0;
};

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
