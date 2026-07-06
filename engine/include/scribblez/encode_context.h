#pragma once

#include "scribblez/input_encoding_spec.h"
#include "scribblez/move.h"

namespace scribblez {

class GameStateEncoder;
class Rack;

// The post-replay state at a sampled position, from which a TrainingTask encodes
// one training row (input tensor + labels). It is the single context object both
// the input encoder and every target read from; a task uses only the fields it
// needs (the max-move-per-lane task ignores the final-score / next-move fields).
struct EncodeContext {
  // Replayed state at the sampled position. `enc` exposes the board, cumulative
  // scores, and both players' last moves; `pov_rack` is the mover's rack (which
  // `enc` does not itself hold). `active_player` is the POV -- the mover at the
  // sampled turn.
  const GameStateEncoder* enc = nullptr;
  const Rack* pov_rack = nullptr;
  int active_player = 0;
  bool apply_flip = false;  // transpose spatial planes/labels across the diagonal

  // The run's input-encoding configuration: the lexicon (which tasks that
  // enumerate legal moves, like max-move-per-lane, also generate with) and
  // which feature blocks the input row carries.
  InputEncodingSpec spec{nullptr, false};

  // Game-outcome context for the post-move labels: each player's next move
  // played from the sampled snapshot onward (opp_next_move: the opponent's
  // upcoming move; self_next_move: the mover's own upcoming move) and the
  // game's final scores. A move past the end of the game leaves the
  // corresponding has_* flag false.
  Move opp_next_move{};
  bool has_opp_next_move = false;
  Move self_next_move{};
  bool has_self_next_move = false;
  int final_score_p0 = 0;
  int final_score_p1 = 0;

  int final_active() const { return active_player == 0 ? final_score_p0 : final_score_p1; }
  int final_opp() const { return active_player == 0 ? final_score_p1 : final_score_p0; }
};

}  // namespace scribblez
