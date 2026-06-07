#pragma once

// BlockDecoder turns a sample-index list (intra-file) into populated rows
// of the DataLoader's training-tensor output buffer.
//
// One instance is owned per decoder worker thread (and, eventually, per
// Python-side handle holding a loaded Dictionary for in-place legal-move
// recomputation at sampled positions). The instance is stateful so that
// the lexicon -- and any other expensive-to-load infrastructure added in
// the future -- need not be reloaded between calls.

#include "scribblez/binary_log.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace binlog {

class BlockDecoder {
 public:
  BlockDecoder() = default;

  // Decode the positions named by `local_indices[0..n_indices)` (each is an
  // intra-file linear index in the .slog file pointed to by `buf`) directly
  // into the row layout starting at `output[output_row_start * kRowFloats]`.
  // `flips[i] != 0` selects diagonal-symmetry augmentation for that row.
  // `post_move` selects whether each row encodes the pre-move snapshot at
  // the sampled turn (false) or the post-move snapshot (true; same player,
  // immediately after their move was applied but before they drew). `path`
  // is used only for diagnostic messages on malformed buffers.
  void decode(const char* buf, const std::string& path, const int64_t* local_indices,
              const uint8_t* flips, std::size_t n_indices, bool post_move, int64_t output_row_start,
              float* output);

 private:
  // One requested sample, post-resolution: which game it lives in, where
  // within that game (intra), and which output row it should land in
  // (out_i, an index into local_indices/flips and thus into output).
  struct Request {
    uint32_t game;
    uint32_t intra;
    uint32_t out_i;
  };

  // Resolve raw local indices into per-game Requests, sorted by (game,
  // intra) so each game is replayed once in forward order. `intra` is
  // simply the turn index k within the game.
  void build_requests(const GameMetadata* metas, uint32_t num_games, const int64_t* local_indices,
                      std::size_t n_indices);

  // Replay one game forward from its initial state, calling emit_row() at
  // every requested turn. `cursor` indexes the next pending Request in
  // requests_; it is advanced past all Requests for this game.
  void replay_game(const char* buf, uint32_t game_idx, const uint8_t* flips, bool post_move,
                   int64_t output_row_start, float* output, std::size_t& cursor);

  // Write one sample row (input encoding + label encoding) from
  // `pov_player`'s POV. Whether the sample is pre-move or post-move is
  // implicit in the current state of enc_ at the call site: pre-move iff
  // enc_.active_player() == pov_player (turn k not yet applied), post-move
  // iff they differ (turn k already applied, enc has flipped active).
  void emit_row(const Request& req, const uint8_t* flips, int64_t output_row_start, float* output,
                int pov_player, const GameMetadata& gm, const TurnBlob* turns);

  // Per-decode buffer reused across calls.
  std::vector<Request> requests_;

  // Per-game working state reset at the top of each replay_game() call.
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

}  // namespace binlog
}  // namespace scribblez
