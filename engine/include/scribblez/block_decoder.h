#pragma once

// BlockDecoder turns a contiguous range of in-file game indices into
// populated rows of the DataLoader's training-tensor output buffer.
//
// One instance is owned per decoder worker thread (and, eventually, per
// Python-side handle holding a loaded Dictionary for in-place legal-move
// recomputation at sampled positions). The instance is stateful so that
// the lexicon -- and any other expensive-to-load infrastructure added in
// the future -- need not be reloaded between calls.

#include "scribblez/game_state_encoder.h"
#include "scribblez/rack.h"

#include <array>
#include <cstdint>
#include <string>

namespace scribblez {
namespace binlog {

class BlockDecoder {
 public:
  BlockDecoder() = default;

  // Decode games [local_start, local_start + n_rows) of the .slog file
  // pointed to by `buf` into output rows [output_row_start, output_row_start
  // + n_rows). One row per game (each game contributes its single
  // pre-chosen sampled turn). `flips[i] != 0` selects diagonal-symmetry
  // augmentation for output row (output_row_start + i). `post_move` picks
  // pre-move vs post-move snapshot at the sampled turn. `path` is used
  // only for diagnostic messages on malformed buffers.
  void decode(const char* buf, const std::string& path, int64_t local_start, int64_t n_rows,
              const uint8_t* flips, bool post_move, int64_t output_row_start, float* output);

 private:
  // Replay one game forward from its initial state up to and (optionally)
  // including its sampled_turn, and emit the resulting row.
  void replay_and_emit(const char* buf, uint32_t game_idx, bool flip, bool post_move,
                       int64_t out_row, float* output);

  // Per-game working state reset at the top of each replay_and_emit() call.
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

}  // namespace binlog
}  // namespace scribblez
