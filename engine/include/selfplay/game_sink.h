#pragma once

#include "game/game.h"

#include <array>

namespace scribblez {

// Destination for finished self-play games. SelfPlayEngine moves each game's
// log storage in, and the sink may retain it (the disk writer) or discard it
// (the streaming encoder). A GameLog view must be taken *after* the move, or
// its pointers dangle.
class GameSink {
 public:
  virtual ~GameSink() = default;

  // `seats[s]` is the persistent player index seated at seat s.
  virtual void on_game(GameLogStorage&& log, const std::array<int, 2>& seats) = 0;
};

}  // namespace scribblez
