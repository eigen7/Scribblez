#pragma once

#include "game/board.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <string>

namespace scribblez {

// The analysis position for the dashboard's lane-analysis tab, parsed from a
// GCG file: the board after all recorded moves, the player to move next, and
// their rack. The rack comes from the file's #RackN pragma, because the
// reader's replayed rack slots cannot tell an empty slot from a hidden tile.
struct GcgAnalysisPosition {
  Board board;
  Rack rack;
  int on_move = 0;
};

// False (with *error set, when non-null) if the GCG does not parse, yields no
// position, or lacks the #RackN pragma for the player on move.
bool parse_gcg_analysis_position(const std::string& gcg_text, GcgAnalysisPosition* out,
                                 std::string* error);

// The web GameState JSON, plus `on_move` and a `lane_analysis` object holding
// each lane's ground-truth targets (training/lane_targets.h) and a sample of
// its maximal plays.
std::string lane_analysis_json(const Board& board, const Rack& rack, int on_move,
                               const Dictionary& dict);

}  // namespace scribblez
