#pragma once

// The DeepBot leaf evaluation training task: a post-move row from the mover's
// side, LeafEvalInputEncoder's input, labelled with the game's outcome, each
// empty square's fate over the rest of the game, which the task declares
// kRecordsSquareFates to receive (encoding/square_fates.h), and how the game
// ended (training/game_end_targets.h). The mover's leave is their rack before
// drawing, and the opponent's is what they kept at their last move, so the row
// is a rollout leaf: the opponent's draw after that move is back in the pool, a
// uniform draw from it like the refill the model averages over. Before the
// opponent's first move their leave is empty and their whole rack is that draw.
//
// The static leave values are HastyBot's (HastyEquity), the rollout policy's.

#include "encoding/encode_context.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "training/game_end_targets.h"
#include "training/square_fate_targets.h"
#include "training/training_targets.h"

namespace scribblez {

struct LeafEvalTask {
  static constexpr bool kRecordsSquareFates = true;
  static constexpr int kInputFloats = LeafEvalInputEncoder::kInputFloats;
  // WldTarget, ScoreDiffTarget, OwnershipTarget, SquarePointsTarget,
  // GoingOutTarget, then EndPointsTarget.
  static constexpr int kWldOffset = 0;
  static constexpr int kScoreDiffOffset = kWldOffset + WldTarget::kDims[0];
  static constexpr int kOwnershipOffset = kScoreDiffOffset + ScoreDiffTarget::kDims[0];
  static constexpr int kSquarePointsOffset = kOwnershipOffset + kBoardCells;
  static constexpr int kGoingOutOffset = kSquarePointsOffset + kBoardCells;
  static constexpr int kEndPointsOffset = kGoingOutOffset + GoingOutTarget::kDims[0];
  static constexpr int kLabelFloats = kEndPointsOffset + EndPointsTarget::kDims[0];
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  // Reads HastyEquity's leave values, which must be initialized.
  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez
