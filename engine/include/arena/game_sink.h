#pragma once

#include "game/game_log.h"

namespace scribblez {

// Destination for finished games. GameEngine moves each game's log storage in,
// and the sink may retain it (the disk writer) or discard it (the streaming
// encoder). A GameLog view must be taken *after* the move, or its pointers
// dangle.
class GameSink {
 public:
  virtual ~GameSink() = default;

  // GameLog::seat_players says which player sat in each seat.
  virtual void on_game(GameLogStorage&& log) = 0;
};

}  // namespace scribblez
