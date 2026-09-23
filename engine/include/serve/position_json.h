#pragma once

// Serializes a board position to the web UI's GameState shape
// (web/src/types.ts). Every producer of GameState JSON builds on
// position_state_object(): the live web server (game_state_json) and the
// offline renderers behind the FFI and tools. It is the engine's one
// definition of that schema. The interactive tools' own state shapes
// (manual_gcg_tool, board_tool) reuse its board, bonus and tile-score fields.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

// 15x15 grid of letters (lowercase for blanks), null for empty squares: the
// `board` field.
boost::json::array board_grid(const Board& board);

// 15x15 grid of premium codes (DL/TL/DW/TW) or null: the `bonuses` field.
boost::json::array bonus_grid(const Board& board);

// Each letter's face value, keyed "A".."Z": the `tile_scores` field.
boost::json::object tile_score_map();

// The tiles in `bag` as {letter, score, count} entries, letters first, then
// the blank as "?"; letters with none left are omitted. The tools' `bag_tiles`
// field.
boost::json::array bag_tiles_json(const TileCounts& bag);

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

// position_state_object_pov() serialized.
std::string position_state_json(const Board& board, const Rack& my_rack, int my_score,
                                int opp_score, const std::string& my_name,
                                const std::string& opp_name);

}  // namespace scribblez
