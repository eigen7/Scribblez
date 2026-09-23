#pragma once

#include "encoding/input_encoder.h"
#include "game/move.h"
#include "game/rack.h"

namespace scribblez {

class GameStateEncoder;

// Everything a TrainingTask may read to encode one row for a replayed
// position: its inputs and its targets. A task uses only the fields it needs.
struct EncodeContext {
  // `enc` holds the replayed public state. `pov_rack` is the POV player's rack,
  // which `enc` does not hold. `opp_known_leave` is what the opponent kept from
  // their last move; only open-leaves specs read it. `active_player` is the
  // POV: the player who moves at the sampled turn.
  const GameStateEncoder* enc = nullptr;
  const Rack* pov_rack = nullptr;
  Rack opp_known_leave{};
  int active_player = 0;

  InputEncodingSpec spec{nullptr};

  // Target data: each player's next move after the sampled position, in
  // `enc`'s board frame (see Board::transpose), and the final scores. A next
  // move past the end of the game leaves its has_* flag false.
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
