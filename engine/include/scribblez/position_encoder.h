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

#include "scribblez/encode_context.h"
#include "scribblez/game.h"
#include "scribblez/game_state_encoder.h"
#include "scribblez/rack.h"

#include <array>

namespace scribblez {

class Dictionary;

namespace binlog {

// The opponent's retained leave at the decision point of `sampled_turn`: their
// current rack (`opp_rack_now`, as tracked by the replay) minus the tiles they
// drew after their most recent move -- i.e. the Bayesian-inferable part of
// their rack, with the fresh replenishment draws masked out. Empty when the
// opponent has not acted yet (their whole rack is an unseen draw). Identical
// for the pre- and post-move snapshots of the same turn (the mover's move does
// not touch the opponent's rack). Serves the open-leaves information condition
// (docs/sim_residual_feedback.md) for both the input encoder and the sim
// tools.
Rack opp_leave_from_replay(const GameLog& g, int sampled_turn, const Rack& opp_rack_now);

class PositionEncoder {
 public:
  // `spec` configures the input encoding (lexicon + feature blocks); tasks
  // that enumerate moves at the position (the max-move-per-lane task)
  // additionally generate with its lexicon.
  explicit PositionEncoder(const InputEncodingSpec& spec) : spec_(spec), enc_(spec) {}

  // Replay `g` from its initial state up to (and, when post_move, including)
  // turn `sampled_turn`. Leaves the internal encoder/racks at that position and
  // returns the POV player (the mover at the sampled turn).
  int replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move);

  // Replay to the sampled position and write one full training row for `Task`
  // (Task::kInputFloats input floats followed by Task::kLabelFloats label
  // floats) to `out_row`. `flip` applies diagonal symmetry to the spatial
  // planes/labels. The replay and context are task-independent; only the final
  // encode differs, so a single encoder serves every task.
  template <typename Task>
  void encode_row(const GameLog& g, int sampled_turn, bool post_move, bool flip, float* out_row);

  // Replay to the sampled position, then encode the input once per integer
  // score differential in [diff_lo, diff_hi], sweeping only the active player's
  // score advantage. Writes (diff_hi - diff_lo + 1) input tensors (kInputFloats
  // each, no labels, no flip) contiguously to `out`. Post-move task only.
  void encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move, int diff_lo,
                               int diff_hi, float* out);

  // Post-replay inspectors (valid after replay_to_sampled / encode_row).
  const GameStateEncoder& enc() const { return enc_; }
  const Rack& rack(int p) const { return racks_[p]; }

 private:
  // Build the EncodeContext for the sampled position after a replay: the
  // replayed state plus the game-outcome fields and the lexicon. `post_move`
  // must match the replay's snapshot kind (it selects which upcoming turn is
  // the mover's own next move). A task reads only the fields it needs.
  EncodeContext make_context(const GameLog& g, int sampled_turn, int mover, bool post_move,
                             bool flip) const;

  InputEncodingSpec spec_;
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

}  // namespace binlog
}  // namespace scribblez

#include "inlines/scribblez/position_encoder.inl"
