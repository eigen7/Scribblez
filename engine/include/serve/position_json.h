#pragma once

// JSON serialization of a board position into the web UI's GameState shape
// (see web/src/types.ts). This is the single source of truth for that schema:
// both the live web server (game_state_json) and the offline position-dump FFI
// build their JSON from position_state_object(), so the two never drift.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

// The GameState object common to every view. Keys are inserted in the order the
// web client expects; callers may append further ones (e.g. moves).
boost::json::object position_state_object(const Board& board, const Rack& my_rack, int my_score,
                                          int opp_score, int bag_size, int opp_rack_size,
                                          const std::string& my_name, const std::string& opp_name,
                                          bool your_turn, bool game_over);

// The same for a static (off-turn) position from the POV player's information
// set, inferring the bag and opponent-rack counts from the standard refill-to-7
// partition.
boost::json::object position_state_object_pov(const Board& board, const Rack& my_rack, int my_score,
                                              int opp_score, const std::string& my_name,
                                              const std::string& opp_name);

// The [row, col] squares a PLAY placed tiles on, empty for EXCHANGE/PASS. The
// web client's last_move shape, which highlights the most recent play.
boost::json::array move_squares(const Move& m);

// position_state_object_pov(...) serialized to a string.
std::string position_state_json(const Board& board, const Rack& my_rack, int my_score,
                                int opp_score, const std::string& my_name,
                                const std::string& opp_name);

}  // namespace scribblez
