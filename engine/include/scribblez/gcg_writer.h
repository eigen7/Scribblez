#pragma once

#include "scribblez/game.h"

#include <ostream>
#include <string>

namespace scribblez {

// Serialize a GameLog as a GCG file -- the de-facto standard Scrabble game-log
// format (as written by Macondo/Quackle). Header pragmata plus one event line
// per turn:
//   >nick: rack POS WORD +score cumulative      (tile placement; '.' = a tile
//                                                 already on the board, lower-
//                                                 case = a designated blank)
//   >nick: rack -TILES +0 cumulative            (exchange)
//   >nick: rack - +0 cumulative                 (pass)
// followed by end-of-game rack adjustments (END_RACK_PTS / END_RACK_PENALTY).
std::string game_log_to_gcg(const GameLog& log);
void write_game_log_gcg(const GameLog& log, std::ostream& out);

}  // namespace scribblez
