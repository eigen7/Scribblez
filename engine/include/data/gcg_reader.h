#pragma once

#include "game/board.h"
#include "game/game.h"
#include "game/rack.h"
#include "game/tile_counts.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {

using ParsedRackSlots = std::array<std::optional<Tile>, RACK_SIZE>;

struct ParsedGcgTurn {
  TurnRecord record;
  std::string notation;
  ParsedRackSlots rack_before_slots;
  std::array<ParsedRackSlots, 2> racks_after_turn;
  std::optional<std::string> exchange_field;
};

struct ParsedGcgSnapshot {
  Board board;
  std::array<int, 2> scores = {0, 0};
  std::array<ParsedRackSlots, 2> racks;
  TileCounts bag;
  int turn_player = 0;
};

// An end-of-game rack adjustment: the player who went out gains the value of
// the opponent's leftover tiles (positive delta); a player left holding tiles
// loses their value (negative delta). `tiles` is the rack scored or penalized.
struct ParsedGcgEndAdjustment {
  int player = 0;
  std::string tiles;
  int delta = 0;
  int total = 0;  // resulting cumulative score for `player`
};

struct ParsedGcgGame {
  std::array<std::string, 2> player_names = {"Player 1", "Player 2"};
  std::vector<ParsedGcgTurn> turns;
  std::vector<ParsedGcgSnapshot> snapshots;
  std::vector<ParsedGcgEndAdjustment> end_adjustments;
  GameLogStorage game_log;

  GameLogStorage to_game_log_storage() const;
};

bool read_gcg_text(const std::string& gcg_text, ParsedGcgGame* out_game,
                   std::string* error_message);

// One endgame position lifted from a GCG's final recorded state: the bag is
// empty, `mover` is to act, and both racks are concrete. racks[mover] comes
// from the file's #RackN pragma (the rack-slot arrays the reader tracks
// cannot distinguish an empty slot from a hidden tile); the other rack is
// derived as the tiles absent from the board and the mover's rack.
struct ParsedGcgEndgame {
  Board board;
  std::array<Rack, 2> racks;
  std::array<int, 2> scores = {0, 0};
  int mover = 0;
  std::array<std::string, 2> player_names;
  int turns = 0;
};

// The rack recorded by a "#RackN TILES" pragma line ('?' is a blank), or
// nullopt when the file has none for that player. The pragma name matches
// case-insensitively, as GCG writers vary.
std::optional<Rack> pragma_rack(const std::string& gcg_text, int player);

// Parse `gcg_text` and lift its final position as an endgame (see
// ParsedGcgEndgame). Returns false with an explanation when the text does not
// parse, the mover's rack pragma is missing, or the position is not a
// bag-empty endgame.
bool read_gcg_endgame(const std::string& gcg_text, ParsedGcgEndgame* out,
                      std::string* error_message);

// The leave `player` retained at their most recent recorded turn: their
// rack_before minus the tiles that move played or exchanged (a PASS retains
// everything). Empty when the player has no recorded turn. This is the
// known part of their rack under the open-leaves information condition --
// their replenishment draws after that move are not recorded and stay hidden.
Rack retained_leave(const ParsedGcgGame& game, int player);

}  // namespace scribblez
