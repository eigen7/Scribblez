#pragma once

// How the position-evaluation datasets read a GCG: the position is the board
// after the final recorded move, seen by the player who made it. That player
// holds only its leave, not having drawn yet. The opponent, to act next, holds
// what its own last move kept plus draws the POV player cannot see.
//
// The Monte-Carlo ground truth (sim/monte_carlo_sim.h) and the model-input
// encoder (training/position_eval_analysis.h) both use this parser, so they
// always agree on the position. gcg_reader.h's read_gcg_position and
// read_gcg_endgame read a file differently: the position before the side to
// move acts, with its rack from a #RackN pragma.

#include "belief/rack_inference.h"
#include "data/gcg_reader.h"
#include "game/board.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <array>
#include <optional>
#include <string>

namespace scribblez {

struct ParsedGcgPostMove {
  ParsedGcgGame game;  // for replaying the moves into an encoder
  Board board;
  std::array<int, 2> scores{0, 0};
  int start_player = 0;  // the player who made the final move: the POV
  Rack leave;            // start_player's rack after the final move
  // What the opponent's last recorded move kept: the known part of their rack
  // under face-up leaves. Empty after a bingo, or if they have no recorded
  // move.
  Rack opp_leave;
  // The opponent's last move as start_player saw it: the board it was played
  // on, the move, and the tiles start_player could not see at the time. A
  // hidden-leaves rollout infers the opponent's leave from this. nullopt if
  // they have no recorded move.
  std::optional<belief::OppMoveObservation> opp_observation;
};

// Returns false and sets `error` when the text does not parse, has no turns,
// or its final move is not a tile placement.
bool read_gcg_post_move(const std::string& gcg_text, ParsedGcgPostMove* out, std::string* error);

// The tiles the holder of `rack` cannot see: the full distribution minus the
// board and `rack`. A designated blank on the board counts as a blank.
TileCounts unseen_counts(const Board& board, const Rack& rack);

}  // namespace scribblez
