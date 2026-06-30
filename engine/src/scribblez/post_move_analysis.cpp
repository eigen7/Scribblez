#include "scribblez/post_move_analysis.h"

#include "scribblez/game_state_encoder.h"
#include "scribblez/gcg_reader.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <cassert>

namespace scribblez {

bool encode_post_move_analysis_input(const std::string& gcg_text, float* out, std::string* error) {
  ParsedGcgGame game;
  if (!read_gcg_text(gcg_text, &game, error)) return false;
  if (game.turns.empty() || game.snapshots.empty()) {
    if (error) *error = "GCG has no turns";
    return false;
  }
  const TurnRecord& last = game.turns.back().record;
  if (last.move.type() != MoveType::PLAY) {
    if (error) *error = "final move is not a tile placement";
    return false;
  }

  // Replay every recorded move into a fresh encoder; apply_move needs only the moves
  // (not racks), so this reproduces the board / scores / last-two-moves state the
  // training replay builds, without any rack bookkeeping.
  GameStateEncoder enc;
  for (const ParsedGcgTurn& t : game.turns) enc.apply_move(t.record.move);

  // The player to act next (the bingoer) is the snapshot's turn_player; the POV is
  // the one that just moved. This matches parse_monte_carlo_position, so the model
  // is evaluated from the same seat the Monte-Carlo ground truth scores.
  const int start_player = 1 - game.snapshots.back().turn_player;
  assert(enc.active_player() == game.snapshots.back().turn_player);

  // start_player's leave = its final rack_before minus the tiles it placed.
  Rack leave = last.rack_before;
  for (int i = 0; i < last.move.num_glyphs(); ++i) leave.remove(last.move.glyph(i).rack_tile());

  enc.encode_input(start_player, leave, /*apply_flip=*/false, out);
  return true;
}

}  // namespace scribblez
