#pragma once

// Serializes a board position to the web UI's GameState shape
// (web/src/types.ts). Every producer of GameState JSON builds on
// position_state_object(): the live web server (game_state_json) and the
// offline renderers behind the FFI and tools. It is the engine's one
// definition of that schema.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

// The GameState fields common to every view. Callers may append more, e.g.
// the move list.
boost::json::object position_state_object(const Board& board, const Rack& my_rack, int my_score,
                                          int opp_score, int bag_size, int opp_rack_size,
                                          const std::string& my_name, const std::string& opp_name,
                                          bool your_turn, bool game_over);

// position_state_object() for a position seen only from the POV player's
// side, with the bag and opponent-rack counts inferred from the unseen tiles:
// the opponent holds up to 7, the bag the rest. Marked as the POV player's
// turn.
boost::json::object position_state_object_pov(const Board& board, const Rack& my_rack, int my_score,
                                              int opp_score, const std::string& my_name,
                                              const std::string& opp_name);

// The [row, col] squares a PLAY placed tiles on, empty for an exchange or
// pass: the web client's last_move shape, used to highlight the latest play.
boost::json::array move_squares(const Move& m);

}  // namespace scribblez
