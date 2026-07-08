#pragma once

#include "encoding/input_encoder.h"

#include <string>

namespace scribblez {

// Encode a penultimate-bingo GCG's analysis position into the position
// evaluation model's input tensor (input_floats(spec) floats), from the POV of
// the player that made the final recorded move (whose leave is the
// encode-time rack). `spec` configures the encoding (lexicon + feature
// blocks).
//
// It replays the recorded moves into a fresh GameStateEncoder, reproducing exactly
// the board / scores / last-two-moves state that the training replay path builds, so
// the encoded input is byte-identical to a training row's input for the same
// position. Returns false (with *error set, if non-null) on a parse error or if the
// final move is not a tile placement.
bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error);

// Like encode_position_eval_analysis_input, but encodes from an explicit
// alternate leave (`leave_str`: A-Z, '?' = a blank) instead of the GCG's
// recorded leave -- a dashboard what-if. The board / scores / moves are
// unchanged; only the rack and unseen-pool features reflect the new leave.
// The alternate leave must have the same number of tiles as the recorded
// leave and use only tiles available off the board. Returns false (with
// *error set) on a parse error, a size mismatch, or an unavailable tile.
bool encode_position_eval_analysis_input_with_leave(const std::string& gcg_text,
                                                    const std::string& leave_str,
                                                    const InputEncodingSpec& spec, float* out,
                                                    std::string* error);

// Build the web-render board bundle for the same analysis position: the
// GameState JSON (board / bonuses / rack / tile_scores, plus a "start_player" field)
// from the POV of the player that made the final move, with its leave as the shown
// rack. Uses the shared position serializer, so the board renders identically to the
// lane-analysis tab. Returns "" (with *error set, if non-null) on a parse error or a
// non-PLAY final move.
std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error);

}  // namespace scribblez
