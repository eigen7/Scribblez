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
  if (game.turns.empty() || game.snapshots.size() != game.turns.size() + 1) {
    if (error) *error = "GCG has no recorded move to take as the decision point";
    return false;
  }
  const size_t last = game.turns.size() - 1;
  const TurnRecord& final_turn = game.turns[last].record;
  out->pos.board = game.snapshots[last].board;
  out->pos.scores = game.snapshots[last].scores;
  out->pos.mover = final_turn.player;
  out->pos.rack = final_turn.rack_before;
  out->pos.opp_leave = open_leaves ? retained_leave(game, 1 - out->pos.mover) : Rack{};
  out->played = final_turn.move;
  out->turn_index = int(last);
  const int pool = unseen_pool(out->pos.board, out->pos.rack, 0).size();
  out->bag_size = std::max(0, pool - RACK_SIZE);
  return true;
}

void replay_to_decision(const GcgDecision& d, GameStateEncoder* enc) {
  for (size_t t = 0; t < d.game.turns.size() - 1; ++t) enc->apply_move(d.game.turns[t].record.move);
  // apply_move attributes each move to the encoder's own turn order (seat 0
  // first), which the recorded seats must follow for scores and last moves to
  // land on the right player.
  RELEASE_ASSERT(enc->active_player() == d.pos.mover);
}

}  // namespace scribblez
