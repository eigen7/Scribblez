#include "game/game_end.h"

namespace scribblez {

namespace {

// Each player's score after the last move, summed from the moves themselves
// as Game::play sums them: a .slog view fills in only each record's player,
// move and draw (binlog::make_game_view).
std::array<int, 2> scores_after_last_move(const GameLog& g) {
  std::array<int, 2> scores = g.initial_scores;
  for (int k = 0; k < g.num_records; ++k) {
    const TurnRecord& rec = g.records[k];
    if (rec.move.type() == MoveType::PLAY) scores[rec.player] += rec.move.score();
  }
  return scores;
}

}  // namespace

GameEnd game_end(const GameLog& g) {
  GameEnd end;
  if (g.num_records == 0) return end;
  const std::array<int, 2> after_last = scores_after_last_move(g);
  for (int p = 0; p < 2; ++p) end.adjustment[p] = g.final_scores[p] - after_last[p];
  // Going out is the only positive adjustment, and the opponent's is then
  // zero; a stalemate charges both players. A game whose leftover tiles were
  // all blanks would adjust nobody and read as unfinished; Game::play asserts
  // that away.
  for (int p = 0; p < 2; ++p) {
    if (end.adjustment[p] > 0) {
      end.kind = GameEndKind::kWentOut;
      end.went_out = p;
      return end;
    }
  }
  if (end.adjustment[0] < 0 || end.adjustment[1] < 0) end.kind = GameEndKind::kStalemate;
  return end;
}

}  // namespace scribblez
