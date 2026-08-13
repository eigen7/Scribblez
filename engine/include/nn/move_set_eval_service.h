#pragma once

#include "nn/eval_service.h"
#include "training/move_set_encoder.h"

// Synchronous move set evaluation: one position's encoder row plus its encoded
// candidate set in, one nn::Eval per candidate out.
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
// scripted stub -- as EvalService already does for the position model. Both
// describe their served model's board rows through ServedModelInputs, so the
// arm-and-width validation an agent does is written once for either.
class MoveSetEvalService : public ServedModelInputs {
 public:
  // `board_row` is the pre-move position, laid out as
  // GameStateEncoder::encode_input() writes it; `moves` are that position's
  // candidates. Writes moves.count Evals to `out`.
  virtual void evaluate(const float* board_row, const move_set::MoveFeatureArrays& moves,
                        Eval* out) = 0;
};

}  // namespace nn
}  // namespace scribblez
