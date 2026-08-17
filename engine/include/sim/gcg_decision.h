// The decision point a .gcg encodes for simulation and evaluation: the state
// before its final recorded move, with the mover's full reconstructed rack.
// The final move itself is not applied -- it is the move that was (or would
// be) played there, kept for reference: the sim tools sim alternatives to it,
// the analysis tools rank it. This is the convention of the hand-maintained
// position sets under positions/ and of the web UI's gcg evidence view.
#pragma once

#include "data/gcg_reader.h"
#include "sim/sim_runner.h"

#include <string>

namespace scribblez {

class GameStateEncoder;

struct GcgDecision {
  ParsedGcgGame game;
  SimPosition pos;  // board, scores, mover, rack; opp_leave set under open leaves
  Move played;      // the final recorded move, not applied
  int turn_index;   // plies before the decision (the final turn's index)
  // The bag from the mover's POV: the unseen pool minus the opponent's
  // (assumed full) rack; 0 in the endgame, which the sims cannot run.
  int bag_size;
};

// Parses `gcg_text` to its decision point. Under `open_leaves` the opponent's
// retained leave -- reconstructable from their last recorded move, empty when
// they bingoed or have not acted -- is set on the position (their
// replenishments stay hidden). False with `*error` set on unparsable text or a
// game with no recorded move.
bool gcg_decision(const std::string& gcg_text, bool open_leaves, GcgDecision* out,
                  std::string* error);

// Replays the moves before the decision into `enc` (a fresh encoder), leaving
// it at the decision point: board, scores, last moves, and the active player
// equal to the mover.
void replay_to_decision(const GcgDecision& d, GameStateEncoder* enc);

}  // namespace scribblez
