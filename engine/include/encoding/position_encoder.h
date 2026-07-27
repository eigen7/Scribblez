#pragma once

// Replays one game -- a GameLog view, from live self-play or reconstructed from
// a .slog buffer -- forward to its sampled position and encodes that position's
// model input and training labels.
//
// The single tensorization path the streaming producer and the on-disk
// DataLoader share, so a row from a live game is byte-identical to one decoded
// from a .slog. Stateful, to reallocate nothing between calls, so each worker
// thread owns one.

#include "encoding/encode_context.h"
#include "encoding/game_state_encoder.h"
#include "game/game.h"
#include "game/rack.h"

#include <array>

namespace scribblez {

class Dictionary;

namespace binlog {

// The opponent's retained leave at `sampled_turn`'s decision point: their
// current rack minus the tiles they drew after their most recent move -- the
// Bayesian-inferable part of their rack, with the fresh draws masked out. Empty
// when the opponent has not acted, their whole rack then being an unseen draw.
// Identical for a turn's pre- and post-move snapshots, the mover's move not
// touching the opponent's rack. Serves the open-leaves information condition
// (docs/sim_residual_feedback.md).
Rack opp_leave_from_replay(const GameLog& g, int sampled_turn, const Rack& opp_rack_now);

class PositionEncoder {
 public:
  explicit PositionEncoder(const InputEncodingSpec& spec) : spec_(spec), enc_(spec) {}

  // Replay `g` up to (and, when post_move, including) turn `sampled_turn`,
  // leaving the encoder and racks there. Returns the POV player.
  int replay_to_sampled(const GameLog& g, int sampled_turn, bool post_move);

  // Replay and write one full training row for `Task` -- Task::kInputFloats
  // input floats then Task::kLabelFloats label floats. `flip` applies the
  // diagonal symmetry. Replay and context are task-independent, only the final
  // encode differs, so one encoder serves every task.
  template <typename Task>
  void encode_row(const GameLog& g, int sampled_turn, bool post_move, bool flip, float* out_row);

  // Replay, then encode the input once per integer score differential in
  // [diff_lo, diff_hi], writing that many input tensors (no labels, no flip)
  // contiguously to `out`. Post-move task only.
  void encode_score_diff_sweep(const GameLog& g, int sampled_turn, bool post_move, int diff_lo,
                               int diff_hi, float* out);

  // Valid after replay_to_sampled / encode_row.
  const GameStateEncoder& enc() const { return enc_; }
  const Rack& rack(int p) const { return racks_[p]; }

 private:
  // The replayed state plus the game-outcome fields and the lexicon; a task
  // reads only what it needs. `post_move` must match the replay's snapshot
  // kind, selecting which upcoming turn is the mover's own next move.
  EncodeContext make_context(const GameLog& g, int sampled_turn, int mover, bool post_move,
                             bool flip) const;

  InputEncodingSpec spec_;
  GameStateEncoder enc_;
  std::array<Rack, 2> racks_{};
};

}  // namespace binlog
}  // namespace scribblez

#include "inlines/encoding/position_encoder.inl"
