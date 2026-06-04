#pragma once
#include <ostream>
#include <string>

#include "scribblez/game.h"

namespace scribblez {

// Serialize a GameLog as JSON (pretty-printed, 2-space indent).
std::string game_log_to_json(const GameLog& log);
void write_game_log_json(const GameLog& log, std::ostream& out);

}  // namespace scribblez
