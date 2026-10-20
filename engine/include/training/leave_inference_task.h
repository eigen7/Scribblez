#pragma once

// The leave inference training task (docs/plans/deep_bot_v0.md, "Rows"): a
// pre-move row from the observer's side, LeaveInferenceInputEncoder's input,
// labelled with the opponent's leave from replay.

#include "encoding/encode_context.h"
#include "encoding/leave_inference_input_encoder.h"
#include "game/rack.h"

#include <cstdint>

namespace scribblez {

// Why a row carries no loss. Its leave's distribution does not depend on the
// model: there is no leave yet, an empty one is certain, and a pass keeps the
// whole rack, drawn from the prior alone. The loader cannot skip these rows,
// since it plans a batch's rows before decoding them, so the loss masks them.
enum class LeaveRowCause : uint8_t {
  kValid = 0,
  kNoOpponentMove = 1,  // turn 0
  kEmptyLeave = 2,      // a bingo, or an exchange of the whole rack
  kPass = 3,
};

// The opponent's leave: its tiles as sorted tile indices padded with -1, its
// length, whether the row carries loss (1 or 0), and the LeaveRowCause. Then
// what the loss and the eval read besides the input: the leave pool
// (leave_pool: the observer's unseen pool, less their exchange just before
// the opponent's move) as per-tile counts, which the leave's prior is over,
// and the bag size.
struct OppLeaveTarget {
  static constexpr int kTiles = RACK_SIZE;
  static constexpr int kLengthOffset = kTiles;
  static constexpr int kValidOffset = kLengthOffset + 1;
  static constexpr int kCauseOffset = kValidOffset + 1;
  static constexpr int kPoolOffset = kCauseOffset + 1;
  static constexpr int kBagOffset = kPoolOffset + TILE_KINDS;
  static constexpr int kFloats = kBagOffset + 1;

  static LeaveRowCause cause(const EncodeContext& ctx);
  static void encode(const EncodeContext& ctx, float* out);
};

struct LeaveInferenceTask {
  // PositionEncoder records the history this task reads.
  static constexpr bool kRecordsLeaveHistory = true;

  static constexpr int kInputFloats = LeaveInferenceInputEncoder::kInputFloats;
  static constexpr int kLabelFloats = OppLeaveTarget::kFloats;
  static constexpr int kRowFloats = kInputFloats + kLabelFloats;

  static void encode_row(const EncodeContext& ctx, float* out_row);
};

}  // namespace scribblez
