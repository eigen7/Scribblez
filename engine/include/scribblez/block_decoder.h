#pragma once

// BlockDecoder turns a contiguous range of in-file game indices into populated
// rows of the DataLoader's training-tensor output buffer. It adapts the on-disk
// .slog format to the shared PositionEncoder: for each game it builds a GameLog
// view over the file bytes (materializing the turn array into a reused scratch
// buffer) and hands that view to the encoder.
//
// One instance is owned per decoder worker thread; it is stateful so the
// encoder and the scratch turn buffer need not be reallocated between calls.

#include "scribblez/game.h"
#include "scribblez/position_encoder.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

class BlockDecoder {
 public:
  BlockDecoder() = default;

  // Decode games [local_start, local_start + n_rows) of the .slog file
  // pointed to by `buf`, emitting each game's eval-only `sampled_turn` into
  // output rows [output_row_start, output_row_start + n_rows). One row per
  // game. `flips[i] != 0` selects diagonal-symmetry augmentation for output
  // row (output_row_start + i). `post_move` picks pre-move vs post-move
  // snapshot. `path` is used only for diagnostic messages. (Training uses
  // decode_one, which addresses an explicit turn; this game-indexed form is
  // for tests / one-position-per-game callers.)
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* flips, bool post_move, int64_t output_row_start, float* output);

  // Decode a single (game, turn) position of the .slog file `buf` into output
  // row `output_row`. This is the training entry point: the DataLoader expands
  // each game into one position per eligible turn and calls this per row.
  // `turn_idx` selects which turn to encode (replacing the file's sampled_turn);
  // `flip` toggles diagonal symmetry; `post_move` picks the snapshot.
  void decode_one(const char* buf, const std::string& path, uint32_t game_idx, uint32_t turn_idx,
                  bool flip, bool post_move, int64_t output_row, float* output);

  // Replay game `game_idx` to its sampled position, then encode that fixed
  // position once per integer score differential in [diff_lo, diff_hi],
  // sweeping only the active player's score advantage. Writes
  // (diff_hi - diff_lo + 1) input tensors (kInputFloats each, no targets, no
  // symmetry flip) contiguously to `out`.
  void encode_score_diff_sweep(const char* buf, uint32_t game_idx, bool post_move, int diff_lo,
                               int diff_hi, float* out);

  // Replay game `game_idx` to its sampled position and render a human-readable
  // ASCII description of that position (POV, scores, leave, last moves, board).
  std::string dump_position(const char* buf, uint32_t game_idx, bool post_move);

  // Replay game `game_idx` to its sampled position and serialize it as the web
  // UI's GameState JSON (board, bonuses, rack, scores, ...) from the POV
  // player's information set. Feeds the web-style image renderer.
  std::string dump_position_json(const char* buf, uint32_t game_idx, bool post_move);

 private:
  // Build a non-owning GameLog view over game `game_idx` of `buf`, filling
  // scratch_ with that game's turns (only `move` and `drawn` -- the fields the
  // encoder reads). Sets *sampled_turn to the game's sampled turn index. The
  // returned view is valid until scratch_ is next reused.
  GameLog game_view(const char* buf, uint32_t game_idx, uint32_t* sampled_turn);

  PositionEncoder pos_;
  std::vector<TurnRecord> scratch_;
};

}  // namespace binlog
}  // namespace scribblez
