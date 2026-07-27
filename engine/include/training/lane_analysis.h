#pragma once

#include "game/board.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <string>

namespace scribblez {

// The analysis position for the lane-analysis UI, parsed from a GCG file: the
// board after all recorded moves, the player to move next, and their rack. The
// move log clears a player's rack as it replays, so the on-move rack comes from
// the GCG's #Rack header rather than the replayed state.
struct GcgAnalysisPosition {
  Board board;
  Rack rack;
  int on_move = 0;
};

// False (with *error set, when non-null) if the GCG has no playable turns or no
// #Rack header for the on-move player.
bool parse_gcg_analysis_position(const std::string& gcg_text, GcgAnalysisPosition* out,
                                 std::string* error);

// The web GameState object the React board renders, extended with `on_move` and
// a `lane_analysis` object holding, per lane, the ground-truth has_move /
// max_score / placed-tile union and the maximal plays.
std::string lane_analysis_json(const Board& board, const Rack& rack, int on_move,
                               const Dictionary& dict);

}  // namespace scribblez
