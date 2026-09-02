#pragma once

// Turns in-file game indices into populated rows of the DataLoader's output
// buffer, adapting the on-disk .slog format to the shared PositionEncoder by
// building a GameLog view over the file bytes per game.
//
// Stateful, to reallocate neither the encoder nor the scratch turn buffer
// between calls, so each decoder worker thread owns one.

#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "game/game_log.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

// Which training task a decoder emits. This fixes the row width and, one level
// up in the DataLoader, which of a game's turns expand into rows:
//   kPositionEval   -- input + wld/score-diff/placement labels; a game expands
//                      over its bag-non-empty eligible-turn prefix, post-move.
//   kMaxMovePerLane -- the lean board/rack input + per-lane labels; a game
//                      expands over every turn, pre-move. The labels come from
//                      enumerating legal moves at the position, so the outcome
//                      and score fields go unused.
enum class DecodeTask { kPositionEval, kMaxMovePerLane };

class BlockDecoder {
 public:
  explicit BlockDecoder(const InputEncodingSpec& spec, DecodeTask task = DecodeTask::kPositionEval)
      : task_(task), row_floats_(row_floats_for(task, spec)), pos_(spec) {}

  // One row per game: each game's eval-only `sampled_turn`, for tests and
  // one-position-per-game callers. `transposes[i] != 0` selects diagonal-symmetry
  // augmentation for output row (output_row_start + i), and `path` appears only
  // in diagnostics. Training instead uses decode_one, which addresses an
  // explicit turn.
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* transposes, bool post_move, int64_t output_row_start, float* output);

  // The training entry point: the DataLoader expands each game into one
  // position per eligible turn and calls this per row.
  void decode_one(const char* buf, const std::string& path, uint32_t game_idx, uint32_t turn_idx,
                  bool transpose, bool post_move, int64_t output_row, float* output);

  // Replay to the game's sampled position, then encode it once per integer
  // score differential in [diff_lo, diff_hi], writing that many input tensors
  // (no targets, natural frame) contiguously to `out`.
  void encode_score_diff_sweep(const char* buf, uint32_t game_idx, bool post_move, int diff_lo,
                               int diff_hi, float* out);

  // The game's sampled position as human-readable ASCII (POV, scores, leave,
  // last moves, board).
  std::string dump_position(const char* buf, uint32_t game_idx, bool post_move);

  // The game's sampled position as the web UI's GameState JSON, from the POV
  // player's information set, for the web-style image renderer.
  std::string dump_position_json(const char* buf, uint32_t game_idx, bool post_move);

 private:
  // Valid until scratch_ is next reused.
  GameLog game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn);

  static int row_floats_for(DecodeTask task, const InputEncodingSpec& spec);

  DecodeTask task_;
  int row_floats_;
  PositionEncoder pos_;
  std::vector<TurnRecord> scratch_;
};

}  // namespace binlog
}  // namespace scribblez
