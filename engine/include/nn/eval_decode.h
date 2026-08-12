#pragma once

#include "nn/eval_service.h"

// The scoring heads both served models carry -- the position evaluation model
// over board rows and the move set evaluation model over candidates -- are the
// same two: 3 raw WLD logits and a score-diff Gaussian. Decoding them into an
// Eval is therefore one function, so the two services cannot drift in how a
// model's raw outputs become a comparable value.

namespace scribblez {
namespace nn {

// `wld_logits` is kWldFloats raw logits [win, draw, loss]; `score_diff` is the
// head's [mean, std] of the final differential, std already positive (softplus
// is applied in-graph). The win-probability scalar treats a draw as half a win
// (expected game points).
Eval decode_eval(const float* wld_logits, const float* score_diff);

}  // namespace nn
}  // namespace scribblez
