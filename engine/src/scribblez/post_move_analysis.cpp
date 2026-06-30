#include "scribblez/post_move_analysis.h"

#include "scribblez/game_state_encoder.h"
#include "scribblez/gcg_reader.h"
#include "scribblez/move.h"
#include "scribblez/position_json.h"
#include "scribblez/rack.h"

#include <boost/json.hpp>

#include <cassert>
#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

// The POV seat and its leave for a parsed post-move position.
struct FinalPosition {
  int start_player = 0;  // seat that made the final move (the evaluated POV)
  Rack leave;            // start_player's leave = final rack_before minus placed tiles
};

// Parse `gcg_text` into `*game` and derive the post-move POV / leave (shared by the
// input encoder and the board bundle, so the two never disagree on the position).
bool parse_final_position(const std::string& gcg_text, ParsedGcgGame* game, FinalPosition* out,
                          std::string* error) {
  if (!read_gcg_text(gcg_text, game, error)) return false;
  if (game->turns.empty() || game->snapshots.empty()) {
    if (error) *error = "GCG has no turns";
    return false;
  }
  const TurnRecord& last = game->turns.back().record;
  if (last.move.type() != MoveType::PLAY) {
    if (error) *error = "final move is not a tile placement";
    return false;
  }
  // turn_player is the seat to act next (the bingoer); the POV is the one that just
  // moved. This matches parse_monte_carlo_position, so the model is evaluated from
  // the same seat the Monte-Carlo ground truth scores.
  out->start_player = 1 - game->snapshots.back().turn_player;
  out->leave = last.rack_before;
  for (int i = 0; i < last.move.num_glyphs(); ++i)
    out->leave.remove(last.move.glyph(i).rack_tile());
  return true;
}

}  // namespace

bool encode_post_move_analysis_input(const std::string& gcg_text, float* out, std::string* error) {
  ParsedGcgGame game;
  FinalPosition pos;
  if (!parse_final_position(gcg_text, &game, &pos, error)) return false;

  // Replay every recorded move into a fresh encoder; apply_move needs only the moves
  // (not racks), so this reproduces the board / scores / last-two-moves state the
  // training replay builds, without any rack bookkeeping.
  GameStateEncoder enc;
  for (const ParsedGcgTurn& t : game.turns) enc.apply_move(t.record.move);
  assert(enc.active_player() == game.snapshots.back().turn_player);

  enc.encode_input(pos.start_player, pos.leave, /*apply_flip=*/false, out);
  return true;
}

std::string post_move_analysis_board_json(const std::string& gcg_text, std::string* error) {
  ParsedGcgGame game;
  FinalPosition pos;
  if (!parse_final_position(gcg_text, &game, &pos, error)) return "";

  const ParsedGcgSnapshot& final_pos = game.snapshots.back();
  const int opp = 1 - pos.start_player;
  const std::string my_name = "Player " + std::to_string(pos.start_player + 1);
  const std::string opp_name = "Player " + std::to_string(opp + 1);
  json::object o =
    position_state_object_pov(final_pos.board, pos.leave, final_pos.scores[pos.start_player],
                              final_pos.scores[opp], my_name, opp_name);
  o["start_player"] = pos.start_player;
  return json::serialize(o);
}

}  // namespace scribblez
