#pragma once

// Decodes positions from an in-memory .slog body into encoded rows, by building
// a GameLog view of the game and handing it to PositionEncoder.
//
// A decoder keeps its encoder and scratch turn buffer across calls to avoid
// reallocating them, so each DataLoader worker thread owns one.

#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "game/game_log.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

// Which training task a decoder emits. This fixes the row layout and, in the
// DataLoader, which of a game's turns become rows:
//   kPositionEval   -- PositionEvalTask rows; one per eligible turn (see
//                      binlog::GameMetadata), pre- or post-move per the
//                      caller's post_move flag.
//   kMaxMovePerLane -- MaxMovePerLaneTask rows; one per turn, always pre-move.
//                      Its labels come from enumerating legal moves at the
//                      position, not from the game's outcome.
enum class DecodeTask { kPositionEval, kMaxMovePerLane };

class BlockDecoder {
 public:
  explicit BlockDecoder(const InputEncodingSpec& spec, DecodeTask task = DecodeTask::kPositionEval)
      : task_(task), row_floats_(row_floats_for(task, spec)), pos_(spec) {}

  // Encodes the sampled_turn of games [local_start, local_start + n_rows) as
  // PositionEvalTask rows, for tests and other one-position-per-game callers.
  // `transposes[i] != 0` transposes output row (output_row_start + i). `path`
  // is used only in error messages.
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* transposes, bool post_move, int64_t output_row_start, float* output);

  // Encodes one row for an explicit turn; the DataLoader's entry point.
  void decode_one(const char* buf, const std::string& path, uint32_t game_idx, uint32_t turn_idx,
                  bool transpose, bool post_move, int64_t output_row, float* output);

  // The board just before turn `turn_idx`'s move, for consumers that derive
  // per-candidate features from it (e.g. move-set cross-check deltas). Valid
  // until this decoder is next used.
  const Board& replay_board(const char* buf, uint32_t game_idx, uint32_t turn_idx);

  // The game's sampled position as human-readable text.
  std::string dump_position(const char* buf, uint32_t game_idx, bool post_move);

  // The game's sampled position as the web UI's GameState JSON, showing only
  // what the player to be evaluated could see.
  std::string dump_position_json(const char* buf, uint32_t game_idx, bool post_move);

 private:
  GameLog game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn);

  static int row_floats_for(DecodeTask task, const InputEncodingSpec& spec);

  DecodeTask task_;
  int row_floats_;
  PositionEncoder pos_;
  std::vector<TurnRecord> scratch_;
};

}  // namespace binlog
}  // namespace scribblez
