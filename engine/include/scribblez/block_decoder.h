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

// Everything needed to score the move played at a game's sampled turn two
// independent ways -- HastyBot static equity and the value model -- so the two
// estimators of the eventual score differential can be compared off-line. All
// fields are taken from the POV of the mover (the player on turn at the sampled
// position). Produced by BlockDecoder::value_probe(); the post-move model input
// is written separately into a caller-provided buffer.
struct ValueProbe {
  int pov = 0;             // mover (player on turn at the sampled position)
  int pre_move_diff = 0;   // mover's score minus opponent's, BEFORE the move
  int final_diff = 0;      // realized final mover-minus-opponent differential
  int bag_size = 0;        // tiles in the bag BEFORE the move (always > 0 here)
  Move played_move;        // the move played at the sampled turn
  Board board;             // board BEFORE the move (for equity scoring)
  Rack mover_rack;         // mover's rack BEFORE the move (yields the leave)
  Rack opp_rack;           // opponent's rack (equity's endgame arg; unused when
                           // bag_size > 0, which always holds at sampled turns)
};

class BlockDecoder {
 public:
  BlockDecoder() = default;

  // Decode games [local_start, local_start + n_rows) of the .slog file pointed
  // to by `buf` into output rows [output_row_start, output_row_start + n_rows).
  // One row per game (each game contributes its single pre-chosen sampled
  // turn). `flips[i] != 0` selects diagonal-symmetry augmentation for output
  // row (output_row_start + i). `post_move` picks pre-move vs post-move snapshot
  // at the sampled turn. `path` is used only for diagnostic messages on
  // malformed buffers.
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* flips, bool post_move, int64_t output_row_start, float* output);

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

  // Replay game `game_idx` to its sampled turn and return the data needed to
  // score the played move both ways (see ValueProbe). The pre-move board / racks
  // / margin feed HastyBot equity; the post-move model input (kInputFloats,
  // no symmetry flip -- exactly what NeuralAgent feeds the model) is written
  // to `post_move_input`.
  ValueProbe value_probe(const char* buf, uint32_t game_idx, float* post_move_input);

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
