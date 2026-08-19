#pragma once

// The position-evaluation datasets' reading of a GCG: the board AFTER the
// final recorded move, evaluated from the POV of the seat that made it. That
// seat holds only its leave (it has not drawn yet); the other seat -- to act
// next -- holds what its own last recorded move retained plus hidden
// replenishments. Both the Monte-Carlo ground truth (sim/monte_carlo_sim.h)
// and the model-input encoder (training/position_eval_analysis.h) read a
// dataset file through this one parser, so they can never disagree on the
// position. read_gcg_endgame / read_gcg_position in gcg_reader.h are the
// other readings (side to move, rack from a #RackN pragma).

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
  int start_player = 0;  // seat that made the final move (the evaluated POV)
  Rack leave;            // start_player's leave = final rack_before minus the placed tiles
  // What the opponent's last recorded move retained -- the known part of their
  // rack under face-up leaves. Empty after a bingo, or when they have no
  // recorded move.
  Rack opp_leave;
  // The opponent's last move as start_player saw it: the board it was played
  // on, the move, and the pool unseen to start_player at that moment (their
  // rack and the bag). The evidence a hidden-leaves rollout infers their leave
  // from; nullopt when they have no recorded move.
  std::optional<belief::OppMoveObservation> opp_observation;
};

// False with an explanation when the text does not parse, has no turns, or
// its final move is not a tile placement.
bool read_gcg_post_move(const std::string& gcg_text, ParsedGcgPostMove* out, std::string* error);

// The full distribution minus the tiles on `board` and in `rack`: everything
// `rack`'s holder cannot see (a designated blank on the board counts as a
// blank).
TileCounts unseen_counts(const Board& board, const Rack& rack);

}  // namespace scribblez
