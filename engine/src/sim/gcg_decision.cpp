#include "sim/gcg_decision.h"

#include "encoding/game_state_encoder.h"
#include "game/rack.h"
#include "util/assert.h"

#include <algorithm>

namespace scribblez {

bool gcg_decision(const std::string& gcg_text, bool open_leaves, GcgDecision* out,
                  std::string* error) {
  if (!read_gcg_text(gcg_text, &out->game, error)) return false;
  const ParsedGcgGame& game = out->game;
  if (game.snapshots.empty()) {
    if (error) *error = "GCG has no position";
    return false;
  }
  const ParsedGcgSnapshot& final_state = game.snapshots.back();
  out->pos.mover = final_state.turn_player;
  const std::optional<Rack> rack = pragma_rack(gcg_text, out->pos.mover);
  if (!rack) {
    if (error) {
      *error = "no #Rack" + std::to_string(out->pos.mover + 1) +
               " pragma: the side to move's rack must be recorded";
    }
    return false;
  }
  out->pos.board = final_state.board;
  out->pos.scores = final_state.scores;
  out->pos.rack = *rack;
  out->pos.opp_leave = open_leaves ? retained_leave(game, 1 - out->pos.mover) : Rack{};
  out->turn_index = int(game.turns.size());
  const int pool = unseen_pool(out->pos.board, out->pos.rack, 0).size();
  out->bag_size = std::max(0, pool - RACK_SIZE);
  return true;
}

void replay_to_decision(const GcgDecision& d, GameStateEncoder* enc) {
  for (const ParsedGcgTurn& t : d.game.turns) enc->apply_move(t.record.move);
  // apply_move attributes each move to the encoder's own turn order (seat 0
  // first), which the recorded seats must follow for scores and last moves to
  // land on the right player.
  RELEASE_ASSERT(enc->active_player() == d.pos.mover);
}

}  // namespace scribblez
