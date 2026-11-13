#pragma once

// An agent's move at a turn of a recorded game: the game is replayed to the
// turn as the agent would have lived it, so an agent that mirrors the game
// (its history, its own earlier racks) holds the state it would hold in play.

#include "agent/agent.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

// A turn of a recorded game, as a positions file names it: a line
// "<.slog path> <game> <turn>", both 0-based.
struct RecordedTurnRef {
  std::string slog;
  uint32_t game = 0;
  int turn = 0;
};

// The turns a positions file names, in its order. Throws util::CleanException
// naming the file and the line it cannot read.
std::vector<RecordedTurnRef> read_recorded_turn_refs(const std::string& path);

// The .slog files `refs` name, by path, each loaded once. Throws
// util::CleanException for a file of another format version, one played with
// face-up leaves (the replay hides the opponent's rack), or a ref to a game
// its file lacks.
std::map<std::string, std::vector<char>> load_recorded_slogs(
  const std::vector<RecordedTurnRef>& refs);

// The decision point before a recorded turn, whole: the replay knows both
// racks.
struct RecordedPosition {
  Board board;
  std::array<Rack, 2> racks;  // by seat
  std::array<int, 2> scores;
  int bag = 0;
  int mover = 0;            // the seat to move
  int scoreless_turns = 0;  // consecutive scoreless turns just before it
};

// The position before turn `turn` of `g`. Reads only each record's move and
// draw.
RecordedPosition recorded_position(const GameLog& g, int turn);

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
