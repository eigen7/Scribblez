#pragma once

#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/rack.h"

#include <string>

namespace scribblez {

// The analysis position for the max-move-per-lane lane-analysis UI, parsed from a
// GCG file: the board after all recorded moves, the player to move next, and that
// player's rack. The move log clears a player's rack as it replays, so the on-move
// rack is read from the GCG's #Rack header rather than the replayed state.
struct GcgAnalysisPosition {
  Board board;
  Rack rack;
  int on_move = 0;
};

// Parse `gcg_text` into its analysis position. Returns false (with *error set, when
// non-null) if the GCG has no playable turns or no #Rack header for the on-move
// player.
bool parse_gcg_analysis_position(const std::string& gcg_text, GcgAnalysisPosition* out,
                                 std::string* error);

// Build the lane-analysis JSON for one (board, rack) position, serialized to a
// string. It is the web GameState object (board, bonuses, rack -- what the React
// board renders) extended with `on_move` and a `lane_analysis` object holding,
// per lane (`rows` and `cols`, 15 each), the ground-truth has_move / max_score /
// placed-tile union and the maximal plays (word, row, col, horizontal, score).
std::string lane_analysis_json(const Board& board, const Rack& rack, int on_move,
                               const Dictionary& dict);

}  // namespace scribblez
