#pragma once

// JSON serialization of a board position into the web UI's GameState shape
// (see web/src/types.ts). This is the single source of truth for that schema:
// both the live web server (game_state_json) and the offline position-dump FFI
// build their JSON from position_state_object(), so the two never drift.

#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

// Build the GameState object common to every view: type, board, bonuses, rack,
// scores, player_names, bag/opponent-rack counts, your_turn, game_over, and the
// per-letter tile_scores map. Keys are inserted in the order the web client and
// existing serializer expect; callers may append further keys (e.g. moves).
boost::json::object position_state_object(const Board& board, const Rack& my_rack, int my_score,
                                          int opp_score, int bag_size, int opp_rack_size,
                                          const std::string& my_name, const std::string& opp_name,
                                          bool your_turn, bool game_over);

// Build the GameState object for a static (off-turn) position from the POV
// player's information set, inferring the bag and opponent-rack counts from the
// standard refill-to-7 partition (unseen = total - board - my_rack). your_turn
// is true and game_over is false. Callers may append further keys (e.g.
// last_move) before serializing.
boost::json::object position_state_object_pov(const Board& board, const Rack& my_rack, int my_score,
                                              int opp_score, const std::string& my_name,
                                              const std::string& opp_name);

// The [row, col] board squares a PLAY placed tiles on (empty for
// EXCHANGE/PASS), derived from the move's lane mask. This is the web client's
// last_move shape, used to highlight the most recent play on the board.
boost::json::array move_squares(const Move& m);

// position_state_object_pov(...) serialized to a string.
std::string position_state_json(const Board& board, const Rack& my_rack, int my_score,
                                int opp_score, const std::string& my_name,
                                const std::string& opp_name);

}  // namespace scribblez
