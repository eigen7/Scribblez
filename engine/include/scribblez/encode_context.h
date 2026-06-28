#pragma once

#include "scribblez/move.h"

namespace scribblez {

class GameStateEncoder;
class Rack;
class Dictionary;

// The post-replay state at a sampled position, from which a TrainingTask encodes
// one training row (input tensor + labels). It is the single context object both
// the input encoder and every target read from; a task uses only the fields it
// needs (the post-move task ignores `dict`; the max-move-per-lane task ignores the
// final-score / next-move fields).
struct EncodeContext {
  // Replayed state at the sampled position. `enc` exposes the board, cumulative
  // scores, and both players' last moves; `pov_rack` is the mover's rack (which
  // `enc` does not itself hold). `active_player` is the POV -- the mover at the
  // sampled turn.
  const GameStateEncoder* enc = nullptr;
  const Rack* pov_rack = nullptr;
  int active_player = 0;
  bool apply_flip = false;  // transpose spatial planes/labels across the diagonal

  // Lexicon, for tasks that enumerate legal moves at the position (the max-move-per-lane
  // task); null for tasks that do not need it.
  const Dictionary* dict = nullptr;

  // Game-outcome context for the post-move labels: the opponent's response to
  // the sampled position and the game's final scores.
  Move next_move{};
  bool has_next_move = false;
  int final_score_p0 = 0;
  int final_score_p1 = 0;

  int final_active() const { return active_player == 0 ? final_score_p0 : final_score_p1; }
  int final_opp() const { return active_player == 0 ? final_score_p1 : final_score_p0; }
};

}  // namespace scribblez
