// The decision point a .gcg encodes for simulation and evaluation: its final
// recorded state, with the side to move holding the rack the file's #RackN
// pragma records. This is the convention of the hand-maintained position
// sets under positions/: a file IS the position to analyze -- every move
// that led there is recorded, nothing after it -- the way read_gcg_endgame
// reads an endgame file, but with a bag.
#pragma once

#include "data/gcg_reader.h"
#include "sim/sim_runner.h"

#include <string>

namespace scribblez {

class GameStateEncoder;

struct GcgDecision {
  ParsedGcgGame game;
  SimPosition pos;  // board, scores, mover, rack; opp_leave set under open leaves
  int turn_index;   // recorded moves before the decision
  // The bag from the mover's POV: the unseen pool minus the opponent's
  // (assumed full) rack; 0 in the endgame, which the sims cannot run.
  int bag_size;
};

// Parses `gcg_text` to its decision point. The mover is the seat to act after
// the last recorded move; its rack must be given by a #RackN pragma (the
// reader's rack-slot arrays cannot tell an empty slot from a hidden tile).
// Under `open_leaves` the opponent's retained leave -- reconstructable from
// their last recorded move, empty when they bingoed or have not acted -- is
// set on the position (their replenishments stay hidden). False with `*error`
// set on unparsable text or a missing rack pragma.
bool gcg_decision(const std::string& gcg_text, bool open_leaves, GcgDecision* out,
                  std::string* error);

// Replays every recorded move into `enc` (a fresh encoder), leaving it at the
// decision point: board, scores, last moves, and the active player equal to
// the mover.
void replay_to_decision(const GcgDecision& d, GameStateEncoder* enc);

}  // namespace scribblez
