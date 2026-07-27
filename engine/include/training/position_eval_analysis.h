#pragma once

#include "encoding/input_encoder.h"

#include <string>

namespace scribblez {

// Encode a penultimate-bingo GCG's analysis position into the position
// evaluation model's input tensor, from the POV of the player that made the
// final recorded move, whose leave is the encode-time rack.
//
// It replays the recorded moves into a fresh GameStateEncoder, reproducing the
// state the training replay path builds, so the input is byte-identical to a
// training row's for the same position. False (with *error set, when non-null)
// on a parse error or a final move that is not a tile placement.
bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error);

// The same from an explicit alternate leave (A-Z, '?' = a blank) rather than
// the GCG's recorded one -- a dashboard what-if, in which only the rack and
// unseen-pool features change. The alternate must hold as many tiles as the
// recorded leave and use only tiles available off the board; a size mismatch or
// an unavailable tile is an error.
bool encode_position_eval_analysis_input_with_leave(const std::string& gcg_text,
                                                    const std::string& leave_str,
                                                    const InputEncodingSpec& spec, float* out,
                                                    std::string* error);

// The web-render bundle for the same analysis position: the GameState JSON plus
// a "start_player" field, with the final mover's leave as the shown rack. It
// goes through the shared position serializer, so the board renders identically
// to the lane-analysis tab. "" (with *error set) on a parse error or a non-PLAY
// final move.
std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error);

}  // namespace scribblez
