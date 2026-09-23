#pragma once

// Replays a game to a chosen turn and encodes a training row (input and
// labels) for the position there. The game is a GameLog, either from live
// self-play or viewed from a .slog buffer.
//
// The streaming producer and the DataLoader both encode through this class,
// so a row from a live game is byte-identical to the same row decoded from a
// .slog. It reuses its buffers between calls, so each worker thread owns one.

#include "encoding/encode_context.h"
#include "encoding/game_state_encoder.h"
#include "game/game_log.h"
#include "game/rack.h"

#include <array>
#include <vector>

namespace scribblez {

class Dictionary;

namespace binlog {

// The leave the opponent of turn `sampled_turn`'s mover kept at their last
// move: `opp_rack_now` minus what they drew after it. This is the part of
// their rack that open leaves makes public. Empty if they have not moved yet.
// The same for the turn's pre- and post-move positions.
Rack opp_leave_from_replay(const GameLog& g, int sampled_turn, const Rack& opp_rack_now);

class PositionEncoder {
 public:
  explicit PositionEncoder(const InputEncodingSpec& spec) : spec_(spec), enc_(spec) {}

  // Replays `g` up to turn `sampled_turn`'s move, or through it (but not its
  // draw) when post_move. Returns the POV player, who makes that move.
  int replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move);

  // Replays, then writes one training row for `Task`. With `transpose`, the
  // replayed state (board, last moves, next moves) is transposed before
  // encoding, so no encoder or target needs to know about the augmentation.
  template <typename Task>
  void encode_row(const GameLog& g, int sampled_turn, bool post_move, bool transpose,
                  float* out_row);

  // Valid after replay_to_sampled / encode_row.
  const GameStateEncoder& enc() const { return enc_; }
  const Rack& rack(int p) const { return racks_[p]; }

  // Tiles in the bag at the replayed position.
  int bag_size() const;

 private:
  // `post_move` must match the replay's, since it decides which later turn is
  // the mover's own next move.
  EncodeContext make_context(const GameLog& g, int sampled_turn, int mover, bool post_move) const;

  InputEncodingSpec spec_;
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

// Writes one post-move input row per candidate, input_floats(spec) floats
// each, for the position `encoder` was replayed to: before turn `turn` of `g`,
// with `mover` (replay_to_sampled's result) to play. `g` is needed for its
// draw records, from which an open-leaves spec gets the opponent's leave.
void encode_candidate_rows(const PositionEncoder& encoder, const GameLog& g, int turn, int mover,
                           const std::vector<Move>& candidates, float* out);

}  // namespace binlog
}  // namespace scribblez

#include "inlines/encoding/position_encoder.inl"
