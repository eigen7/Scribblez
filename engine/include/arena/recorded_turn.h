#pragma once

// An agent's move at a turn of a recorded game: the game is replayed to the
// turn as the agent would have lived it, so an agent that mirrors the game
// (its history, its own earlier racks) holds the state it would hold in play.

#include "agent/agent.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"

namespace scribblez {

class Dictionary;

// The decision point before a recorded turn, and what the agent chose there.
struct RecordedPick {
  Board board;  // before the move
  Rack rack;    // the mover's
  int score = 0;
  int opp_score = 0;
  int bag = 0;
  Move pick;
};

// `agent`'s move at turn `turn` of `g`, in the seat that moved there. The
// agent is told the game began, sees every earlier move (Agent::observe_move)
// and, at each earlier turn of its seat, the request it would have had
// (Agent::observe_turn), and is then asked. Hidden leaves: a request names the
// opponent's rack only with the bag empty. Reads only each record's move and
// draw, as a .slog view fills them (binlog::make_game_view).
RecordedPick pick_at_recorded_turn(Agent& agent, const GameLog& g, int turn,
                                   const Dictionary& dict);

}  // namespace scribblez
