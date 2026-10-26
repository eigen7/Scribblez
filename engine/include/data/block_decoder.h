#pragma once

// Decodes positions from an in-memory .slog body into encoded rows, by building
// a GameLog view of the game and handing it to PositionEncoder.
//
// A decoder keeps its encoder and scratch turn buffer across calls to avoid
// reallocating them, so each DataLoader worker thread owns one.

#include "data/binary_log.h"
#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "game/game_log.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

class ForgoneSidecar;

namespace binlog {

// Which training task a decoder emits. This fixes the row layout and, in the
// DataLoader, which of a game's turns become rows:
//   kPositionEval   -- PositionEvalTask rows; one per eligible turn (see
//                      binlog::GameMetadata), pre- or post-move per the
//                      caller's post_move flag.
//   kMaxMovePerLane -- MaxMovePerLaneTask rows; one per turn, always pre-move.
//                      Its labels come from enumerating legal moves at the
//                      position, not from the game's outcome.
//   kLeaveInference -- LeaveInferenceTask rows; one per eligible turn, always
//                      pre-move. Its labels are the opponent's leave.
//   kPublicLeaveInference -- PublicLeaveInferenceTask rows: kLeaveInference's
//                      rows and labels, the input in its public view.
//   kLeafEval       -- LeafEvalTask rows; one per eligible turn after which
//                      the bag still holds tiles, always post-move. Its
//                      labels are the game's outcome.
enum class DecodeTask {
  kPositionEval,
  kMaxMovePerLane,
  kLeaveInference,
  kPublicLeaveInference,
  kLeafEval
};

class BlockDecoder {
 public:
  explicit BlockDecoder(const InputEncodingSpec& spec, DecodeTask task = DecodeTask::kPositionEval)
      : task_(task), row_floats_(row_floats_for(task, spec)), pos_(spec) {}

  // A `task` row's float counts: its input, its labels, and both.
  static int input_floats_for(DecodeTask task, const InputEncodingSpec& spec);
  static int label_floats_for(DecodeTask task);
  static int row_floats_for(DecodeTask task, const InputEncodingSpec& spec) {
    return input_floats_for(task, spec) + label_floats_for(task);
  }

  // Encodes the sampled_turn of games [local_start, local_start + n_rows) as
  // PositionEvalTask rows, for tests and other one-position-per-game callers.
  // `transposes[i] != 0` transposes output row (output_row_start + i). `path`
  // is used only in error messages.
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* transposes, bool post_move, int64_t output_row_start, float* output);

  // The turns of a game that become `task` rows, [begin, end): every turn for
  // kMaxMovePerLane; for kLeafEval, the eligible turns after which the bag
  // still holds tiles, since a DeepBot rollout whose two plies empty the bag
  // plays to the game's end instead of stopping at a leaf; the eligible region
  // (GameMetadata) otherwise.
  static EligibleSpan row_turns(DecodeTask task, const GameMetadata& meta);

  // Whether `task` rows read the .slog's .forgone sidecar.
  static bool reads_forgone(DecodeTask task);

  // Encodes one row for an explicit turn; the DataLoader's entry point.
  // `forgone` is the .slog's sidecar, required iff reads_forgone(task). Throws
  // util::Exception for a leave inference turn outside the game's eligible
  // region, whose history the sidecar does not cover.
  void decode_one(const char* buf, const std::string& path, uint32_t game_idx, uint32_t turn_idx,
                  RowAugment augment, bool post_move, int64_t output_row, float* output,
                  const ForgoneSidecar* forgone = nullptr);

  // The board decode_one last encoded, after its transpose, with its
  // move-generation caches built: for consumers that derive per-candidate
  // features from the same replay (e.g. move-set cross-checks). Valid until
  // this decoder is next used.
  const Board& board() const { return pos_.enc().board(); }

  // The replay decode_one last ran, for consumers that read its racks or bag
  // as well (e.g. static equities at the decision point). Valid until this
  // decoder is next used.
  const PositionEncoder& position() const { return pos_; }

  // The board just before turn `turn_idx`'s move, replayed without encoding a
  // row, for consumers that need only the board (e.g. reply blocking). Valid
  // until this decoder is next used.
  const Board& replay_board(const char* buf, uint32_t game_idx, uint32_t turn_idx);

 private:
  GameLog game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn);

  // decode_one for a leave inference `Task`.
  template <typename Task>
  void decode_leave_row(const char* buf, const std::string& path, const GameLog& g,
                        uint32_t game_idx, uint32_t turn_idx, RowAugment augment, float* out,
                        const ForgoneSidecar& forgone);

  DecodeTask task_;
  int row_floats_;
  PositionEncoder pos_;
  std::vector<TurnRecord> scratch_;
};

}  // namespace binlog
}  // namespace scribblez
