#pragma once

// PositionEncoder replays one game -- supplied as a GameLog view, whether from
// live self-play or reconstructed from a .slog buffer -- forward to its sampled
// position and encodes the model input and training labels for that position.
//
// It is the single tensorization path shared by the streaming self-play
// producer and the on-disk DataLoader, so a row produced from a live game is
// byte-identical to one decoded from a .slog file. The instance is stateful (it
// holds a GameStateEncoder and the two replayed racks) so nothing need be
// reallocated between calls; one instance is owned per worker thread.

#include "scribblez/game.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/rack.h"

#include <array>

namespace scribblez {
namespace binlog {

class PositionEncoder {
 public:
  PositionEncoder() = default;

  // Replay `g` from its initial state up to (and, when post_move, including)
  // turn `sampled_turn`. Leaves the internal encoder/racks at that position and
  // returns the POV player (the mover at the sampled turn).
  int replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move);

  // Replay to the sampled position and write one full training row
  // (kInputFloats input floats followed by kLabelFloats label floats) to
  // `out_row`. `flip` applies diagonal symmetry to the spatial planes/labels.
  void encode_row(const GameLog& g, int sampled_turn, bool post_move, bool flip, float* out_row);

  // Replay to the sampled position, then encode the input once per integer
  // score differential in [diff_lo, diff_hi], sweeping only the active player's
  // score advantage. Writes (diff_hi - diff_lo + 1) input tensors (kInputFloats
  // each, no labels, no flip) contiguously to `out`.
  void encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move, int diff_lo,
                               int diff_hi, float* out);

  // Post-replay inspectors (valid after replay_to_sampled / encode_row).
  const GameStateEncoder& enc() const { return enc_; }
  const Rack& rack(int p) const { return racks_[p]; }

 private:
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

}  // namespace binlog
}  // namespace scribblez
