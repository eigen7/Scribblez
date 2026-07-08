#pragma once

#include "game/game.h"

#include <array>

namespace scribblez {

// Destination for finished self-play games. SelfPlayEngine hands ownership of
// each game's log storage to on_game (a move, no copy); the sink derives a
// GameLog view from it as needed -- taken *after* the move so the view's
// pointers are valid -- and may retain (e.g. the disk writer) or discard it
// (e.g. the streaming encoder).
class GameSink {
 public:
  virtual ~GameSink() = default;

  // `seats[s]` is the persistent player index (0 or 1) seated at seat s.
  virtual void on_game(GameLogStorage&& log, const std::array<int, 2>& seats) = 0;
};

}  // namespace scribblez
