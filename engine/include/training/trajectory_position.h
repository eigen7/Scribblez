#pragma once

#include "data/gcg_reader.h"
#include "encoding/input_encoder.h"
#include "game/move.h"

#include <string>
#include <vector>

namespace scribblez {

// A position-set .gcg's decision point (read_gcg_position: the final recorded
// state, the side to move next holding its #RackN rack) as the move set
// evaluation model and the trajectory generator see it. The dashboard's
// trajectory pane re-scores this decision under a torch checkpoint, so it needs
// exactly the inputs TrajectoryRunner::run built when the position's .sobs was
// simmed: the mover's pre-move board row, the pre-move score differential the
// move encoder forms resultant differentials from, and the full legal move
// list in the equity ranking the generator drew its candidates from.
struct TrajectoryDecision {
  ParsedGcgPosition position;
  std::vector<Move> legal_moves;  // equity_top_k with no cap; the .sobs candidates are among them
};

// Parse and rank. `open_leaves` selects the information condition (the
// opponent's retained leave is public, and enters the equity ranking); it
// must be the condition the position's sidecars were simmed under. Requires
// HastyEquity to be initialized. False with an explanation on a parse error
// or a missing rack pragma.
bool read_trajectory_decision(const std::string& gcg_text, const Dictionary& dict, bool open_leaves,
                              TrajectoryDecision* out, std::string* error);

// The mover's pre-move board input row under `spec` (input_floats(spec) floats
// into `out`, no symmetry flip) and the pre-move score differential. Replays
// the recorded moves into a fresh GameStateEncoder, as the generator does, so
// the row is the one the student was trained on for this position.
// spec.opp_leave_input must agree with the open_leaves the decision was read
// under (the leave is encoded only when both hold).
void encode_trajectory_decision(const TrajectoryDecision& d, const InputEncodingSpec& spec,
                                float* out, int* score_diff);

// The web-render bundle for the pane: the GameState JSON from the mover's POV
// (board / bonuses / rack / scores / bag and opponent-rack counts /
// tile_scores) plus "mover", "opp_leave" (the known part of the opponent's
// rack, "" when hidden), "last_move" (the squares of the most recent recorded
// play) and "moves": every legal move's GCG notation, in legal_moves order.
std::string trajectory_decision_board_json(const TrajectoryDecision& d);

}  // namespace scribblez
