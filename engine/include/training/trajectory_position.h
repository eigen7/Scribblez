#pragma once

#include "data/gcg_reader.h"
#include "encoding/input_encoder.h"
#include "game/move.h"

#include <string>
#include <vector>

namespace scribblez {

// A position-set .gcg's decision point (read_gcg_position: the final recorded
// state, with the side to move holding its #RackN rack), for the dashboard's
// Trajectories tab. The tab re-scores the decision under a torch checkpoint, so
// it needs exactly the inputs TrajectoryRunner::run built when the position's
// .sobs was simmed: the mover's pre-move board row, the pre-move score
// differential, and the full legal move list in the equity ranking the
// candidates were drawn from.
struct TrajectoryDecision {
  ParsedGcgPosition position;
  std::vector<Move> legal_moves;  // equity_top_k, uncapped; includes the .sobs candidates
};

// Parse and rank. `open_leaves` (the opponent's retained leave is public and
// enters the equity ranking) must match the condition the position's sidecars
// were simmed under. Requires HastyEquity to be initialized. False with an
// explanation on a parse error or a missing rack pragma.
bool read_trajectory_decision(const std::string& gcg_text, const Dictionary& dict, bool open_leaves,
                              TrajectoryDecision* out, std::string* error);

// The mover's pre-move board input row under `spec` (input_floats(spec) floats,
// untransposed) and the pre-move score differential. The replay matches the
// generator's, so the row is the one the student saw for this position.
// spec.opp_leave_input must agree with the open_leaves the decision was read
// under.
void encode_trajectory_decision(const TrajectoryDecision& d, const InputEncodingSpec& spec,
                                float* out, int* score_diff);

// The web-render bundle for the tab: the GameState JSON from the mover's POV,
// plus "mover", "opp_leave" (the known part of the opponent's rack, "" when
// hidden), "last_move" (the squares of the last recorded move) and "moves"
// (every legal move's GCG notation, in legal_moves order).
std::string trajectory_decision_board_json(const TrajectoryDecision& d);

}  // namespace scribblez
