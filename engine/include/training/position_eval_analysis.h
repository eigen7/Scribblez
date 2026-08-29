#pragma once

#include "encoding/input_encoder.h"

#include <string>

namespace scribblez {

// Encode a dataset GCG's post-move position (data/gcg_post_move.h) into the
// position evaluation model's input tensor, from the POV of the player that
// made the final recorded move, whose leave is the encode-time rack; under the
// opponent-leave arm the opponent's retained leave is encoded too.
//
// It replays the recorded moves into a fresh GameStateEncoder, reproducing the
// state the training replay path builds, so the input is byte-identical to a
// training row's for the same position. False (with *error set, when non-null)
// on a parse error or a final move that is not a tile placement.
bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error);

// The same from explicit alternate leaves (A-Z, '?' = a blank) rather than the
// GCG's recorded ones -- a dashboard what-if, in which only the rack,
// opponent-leave, and unseen-pool features change. `opp_leave_str` may be null
// to keep the recorded opponent leave. Each alternate must hold as many tiles
// as the leave it replaces, and the two together may use only tiles available
// off the board; a size mismatch or an unavailable tile is an error.
bool encode_position_eval_analysis_input_with_leaves(const std::string& gcg_text,
                                                     const std::string& leave_str,
                                                     const std::string* opp_leave_str,
                                                     const InputEncodingSpec& spec, float* out,
                                                     std::string* error);

// Collapse the four placement heads' raw footprint logits at this analysis
// position into the four board-frame per-cell occupancy marginals the dashboard
// overlay and Monte-Carlo truth compare against -- the same
// mask+softmax+scatter the .mset writer applies (training/footprint_collapse.h),
// run here on the analysis (post-move) board so the web endpoint keeps reading a
// (15,15) plane per head. `raw` is kPlacementHeads * kFootprintClasses (the
// model's undecoded output); `out` receives kPlacementHeads * (side*side)
// floats. False (with *error) on a parse error or a non-PLAY final move.
bool collapse_position_eval_analysis_placement(const std::string& gcg_text,
                                               const InputEncodingSpec& spec, const float* raw,
                                               float* out, std::string* error);

// The web-render bundle for the same analysis position: the GameState JSON plus
// "start_player", "last_move", and "opp_leave" (the opponent's retained leave,
// '?' = a blank), with the final mover's leave as the shown rack. It goes
// through the shared position serializer, so the board renders identically to
// the lane-analysis tab. "" (with *error set) on a parse error or a non-PLAY
// final move.
std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error);

}  // namespace scribblez
