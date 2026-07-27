#pragma once

#include "encoding/input_encoder.h"
#include "game/move.h"
#include "game/rack.h"

namespace scribblez {

class GameStateEncoder;

// The post-replay state at a sampled position, from which a TrainingTask encodes
// one training row. The single context object both the input encoder and every
// target read from; a task uses only the fields it needs.
struct EncodeContext {
  // `enc` exposes the board, cumulative scores, and both players' last moves;
  // `pov_rack` is the mover's rack, which `enc` does not itself hold.
  // `opp_known_leave` is what the opponent retained from their last move, read
  // only under the open-leaves arm and empty when they have not acted or kept
  // nothing. `active_player` is the POV -- the mover at the sampled turn.
  const GameStateEncoder* enc = nullptr;
  const Rack* pov_rack = nullptr;
  Rack opp_known_leave{};
  int active_player = 0;
  bool apply_flip = false;  // transpose spatial planes/labels across the diagonal

  InputEncodingSpec spec{nullptr, false};

  // Each player's next move from the sampled snapshot onward, and the game's
  // final scores. A move past the end of the game leaves its has_* flag false.
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
