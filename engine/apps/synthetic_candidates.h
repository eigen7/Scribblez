#pragma once

// The synthetic candidate set mset_infer_smoke and proposal_infer_smoke score.
// Sharing it keeps the two tools' outputs comparable.

#include "training/move_set_encoder.h"

namespace scribblez::move_set {

// Candidates of every shape the model sees: plays of 1..7 tiles spread across
// the board, and every fifth one an exchange, which has tiles but no squares.
MoveFeatureArrays synthetic_candidates(int num_moves);

}  // namespace scribblez::move_set
