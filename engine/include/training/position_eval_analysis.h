#pragma once

#include "encoding/input_encoder.h"

#include <string>

namespace scribblez {

// Dashboard analysis of a dataset GCG's post-move position (data/gcg_post_move.h)
// under the position evaluation model. Every function here takes the GCG text
// and returns false (with *error set, when non-null) on a parse error or a final
// move that is not a tile placement.

// Encode the position into the model's input tensor, from the POV of the player
// who made the final recorded move, holding their leave; under the
// opponent-leave arm the opponent's retained leave is encoded too. The recorded
// moves are replayed into a fresh GameStateEncoder, as the training replay path
// does, so the input is byte-identical to a training row's for the same
// position.
bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error);

// The same with alternate leaves (A-Z, '?' = a blank) in place of the recorded
// ones: a what-if in which only the rack, opponent-leave and unseen-pool
// features change. A null `opp_leave_str` keeps the recorded opponent leave.
// Each alternate must hold as many tiles as the leave it replaces, and the two
// together may use only tiles available off the board.
bool encode_position_eval_analysis_input_with_leaves(const std::string& gcg_text,
                                                     const std::string& leave_str,
                                                     const std::string* opp_leave_str,
                                                     const InputEncodingSpec& spec, float* out,
                                                     std::string* error);

// collapse_footprint_planes (training/footprint_collapse.h) at this position:
// the per-cell occupancy planes the dashboard overlay shows and compares against
// Monte Carlo truth. Availability is the unseen pool from the final mover's POV.
// `raw` is kPlacementHeads * kFootprintClasses undecoded logits; `out` receives
// kPlacementHeads * kFootprintCells floats.
bool collapse_position_eval_analysis_placement(const std::string& gcg_text,
                                               const InputEncodingSpec& spec, const float* raw,
                                               float* out, std::string* error);

// masked_placement_distributions at this position, under the same availability:
// `out` receives kPlacementHeads * kFootprintClasses floats. Used to measure how
// sparse the teacher's footprint distributions are.
bool masked_position_eval_analysis_placement(const std::string& gcg_text,
                                             const InputEncodingSpec& spec, const float* raw,
                                             float* out, std::string* error);

// collapse_footprint_legal_cells at this position, under the same availability.
bool legal_position_eval_analysis_placement(const std::string& gcg_text,
                                            const InputEncodingSpec& spec, float* out,
                                            std::string* error);

// The web-render bundle for the position: the GameState JSON, with the final
// mover's leave as the shown rack, plus "start_player", "last_move", and
// "opp_leave" (the opponent's retained leave, '?' = a blank). Returns "" where
// the others return false.
std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error);

}  // namespace scribblez
