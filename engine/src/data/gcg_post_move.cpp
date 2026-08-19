#include "data/gcg_post_move.h"

#include "game/glyph.h"
#include "game/move.h"
#include "game/tile.h"
#include "util/assert.h"

namespace scribblez {

namespace {

// `rack` minus the tiles `move` placed or exchanged (a PASS removes nothing).
Rack leave_after(const Rack& rack, const Move& move) {
  Rack leave = rack;
  for (int i = 0; i < move.num_glyphs(); ++i) {
    const bool ok = leave.remove(move.glyph(i).rack_tile());
    RELEASE_ASSERT(ok, "a move plays tiles its rack does not hold");
  }
  return leave;
}

// The opponent's last recorded turn, seen from start_player's seat at that
// moment. The turns alternate, so when the opponent has a turn at all it is
// the penultimate one, and the rack start_player held while it was played is
// the final turn's rack_before (start_player drew after its previous move and
// before the opponent's).
std::optional<belief::OppMoveObservation> observe_opp_move(const ParsedGcgGame& game,
                                                           int start_player) {
  const int n = int(game.turns.size());
  if (n < 2 || game.turns[n - 2].record.player == start_player) return std::nullopt;
  RELEASE_ASSERT(game.snapshots.size() == game.turns.size() + 1);
  const ParsedGcgTurn& opp_turn = game.turns[n - 2];
  return belief::OppMoveObservation{
    game.snapshots[n - 2].board, opp_turn.record.move,
    unseen_counts(game.snapshots[n - 2].board, game.turns[n - 1].record.rack_before)};
}

}  // namespace

bool read_gcg_post_move(const std::string& gcg_text, ParsedGcgPostMove* out, std::string* error) {
  if (!read_gcg_text(gcg_text, &out->game, error)) return false;
  const ParsedGcgGame& game = out->game;
  if (game.turns.empty() || game.snapshots.empty()) {
    if (error) *error = "GCG has no turns";
    return false;
  }
  const TurnRecord& last = game.turns.back().record;
  if (last.move.type() != MoveType::PLAY) {
    if (error) *error = "final move is not a tile placement";
    return false;
  }
  const ParsedGcgSnapshot& final_pos = game.snapshots.back();
  out->board = final_pos.board;
  out->scores = final_pos.scores;
  // turn_player is the seat to act next; the POV is the one that just moved.
  out->start_player = 1 - final_pos.turn_player;
  out->leave = leave_after(last.rack_before, last.move);
  out->opp_leave = retained_leave(game, 1 - out->start_player);
  out->opp_observation = observe_opp_move(game, out->start_player);
  return true;
}

TileCounts unseen_counts(const Board& board, const Rack& rack) {
  TileCounts unseen;
  for (int t = 0; t < 27; ++t)
    for (int k = 0; k < TILE_COUNTS[t]; ++k) unseen.add(Tile::of(t));
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      const bool ok = unseen.remove(g.is_blank() ? BLANK : g.letter());
      RELEASE_ASSERT(ok, "the board holds more of a tile than the distribution has");
    }
  for (int i = 0; i < rack.size(); ++i) {
    const bool ok = unseen.remove(rack.tiles()[i]);
    RELEASE_ASSERT(ok, "the rack holds a tile the board already exhausted");
  }
  return unseen;
}

}  // namespace scribblez
