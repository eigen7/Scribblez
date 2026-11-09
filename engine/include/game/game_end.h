#pragma once

// How a game ended, read from the gap between its final scores and the scores
// after its last move. A GameLog decoded from a .slog carries neither the end
// reason nor the final racks, so this is the one derivation every reader of
// the end-of-game adjustments uses (the GCG writer, the training targets).
//
// Game::play applies the tournament convention: a player who goes out gains
// twice the opponent's leftover tile values and the opponent loses nothing; in
// a stalemate (six scoreless turns) each player loses their own. A game cut
// short (Game::set_max_plies, or the turn cap) adjusts nobody.

#include "game/game_log.h"

#include <array>

namespace scribblez {

enum class GameEndKind {
  kUnfinished,  // no adjustment: cut short, or no moves at all
  kWentOut,
  kStalemate,
};

struct GameEnd {
  GameEndKind kind = GameEndKind::kUnfinished;
  int went_out = -1;  // the player who went out, for kWentOut
  // Each player's end-of-game adjustment: final score minus the score after
  // the last move.
  std::array<int, 2> adjustment = {0, 0};

  // The net adjustment from `player`'s side: theirs minus the opponent's.
  int net(int player) const { return adjustment[player] - adjustment[1 - player]; }
};

GameEnd game_end(const GameLog& g);

}  // namespace scribblez
