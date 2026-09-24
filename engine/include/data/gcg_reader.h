#pragma once

#include "game/board.h"
#include "game/game_log.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// Parses GCG game records, the Scrabble game-log format written by Macondo,
// Quackle and Woogles, and extracts single positions from them. The position
// readers here take the side to move with its rack from a #RackN pragma;
// gcg_post_move.h has the post-move reading the evaluation datasets use.

namespace scribblez {

struct ParsedGcgTurn {
  TurnRecord record;
  std::string notation;
  std::array<Rack, 2> racks_after_turn;
  std::optional<std::string> exchange_field;
};

struct ParsedGcgSnapshot {
  Board board;
  std::array<int, 2> scores = {0, 0};
  std::array<Rack, 2> racks;
  TileCounts bag;
  int turn_player = 0;
};

// An end-of-game rack adjustment: the player who went out gains the value of
// the opponent's leftover tiles; a player left holding tiles loses theirs.
struct ParsedGcgEndAdjustment {
  int player = 0;
  std::string tiles;
  int delta = 0;
  int total = 0;  // resulting cumulative score for `player`
};

// snapshots[i] is the state before turns[i]; the last snapshot is the state
// after the final turn, with end-of-game adjustments applied to its scores.
struct ParsedGcgGame {
  std::array<std::string, 2> player_names = {"Player 1", "Player 2"};
  std::vector<ParsedGcgTurn> turns;
  std::vector<ParsedGcgSnapshot> snapshots;
  std::vector<ParsedGcgEndAdjustment> end_adjustments;
  // Each player's "#RackN TILES" pragma from the header (before any event
  // line), which gives their rack in the final recorded position; nullopt if
  // the header has none, which is not the same as an empty rack. Only the
  // capitalized "#Rack" form gcg_writer.h emits is a pragma. The reader already
  // applies these to snapshots.back().
  std::array<std::optional<Rack>, 2> header_racks;
  GameLogStorage game_log;  // to_game_log_storage() of the above

  GameLogStorage to_game_log_storage() const;
};

// Returns false and sets `error_message` when the text has no parseable turn.
// Lines it cannot parse are skipped.
bool read_gcg_text(const std::string& gcg_text, ParsedGcgGame* out_game,
                   std::string* error_message);

// The endgame position at a GCG's final recorded state: the bag is empty,
// `mover` is to act, and both racks are known. racks[mover] comes from the
// file's #RackN pragma, because a rack field's '_' cannot tell an empty slot
// from a hidden tile. The other rack comes from its own pragma if present, else it is
// whatever the board and the mover's rack leave unaccounted for.
struct ParsedGcgEndgame {
  Board board;
  std::array<Rack, 2> racks;
  std::array<int, 2> scores = {0, 0};
  int mover = 0;
  std::array<std::string, 2> player_names;
  int turns = 0;
};

// Returns false and sets `error_message` when the text does not parse, the
// mover's rack pragma is missing, or the opponent's rack must be inferred
// and the position is not a bag-empty endgame.
bool read_gcg_endgame(const std::string& gcg_text, ParsedGcgEndgame* out,
                      std::string* error_message);

// The position at a GCG's final recorded state, with `mover` to act holding
// the rack from the file's #RackN pragma. This is how the hand-maintained
// position sets under positions/ are read: a file records every move leading
// to the position to analyze and nothing after it.
struct ParsedGcgPosition {
  ParsedGcgGame game;  // for replaying the moves into an encoder
  Board board;
  std::array<int, 2> scores = {0, 0};
  int mover = 0;
  Rack rack;  // the mover's full rack
  // With open leaves, the opponent's retained_leave(); otherwise empty.
  Rack opp_leave;
  int turns = 0;  // recorded moves before the position
  // The tiles the mover cannot see, minus a full opponent rack; 0 once that
  // goes negative.
  int bag_size = 0;
};

// Returns false and sets `error_message` when the text does not parse or the
// mover's rack pragma is missing.
bool read_gcg_position(const std::string& gcg_text, bool open_leaves, ParsedGcgPosition* out,
                       std::string* error_message);

// Like read_gcg_position, but for the position before recorded turn
// `turn_index` (0-based), so any turn of a recorded game can be analyzed
// without truncating the file by hand. The mover's rack comes from that turn's
// line, and out->game is cut to the turns before it. Returns false and sets
// `error_message` when the text does not parse or `turn_index` is out of
// range.
bool read_gcg_position_at(const std::string& gcg_text, int turn_index, bool open_leaves,
                          ParsedGcgPosition* out, std::string* error_message);

// The leave `player` kept at their most recent recorded turn: that turn's
// rack_before minus the tiles played or exchanged. Empty if they have no
// recorded turn. Under open leaves this is the known part of their rack; their
// draws since are hidden.
Rack retained_leave(const ParsedGcgGame& game, int player);

}  // namespace scribblez
