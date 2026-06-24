#pragma once

#include "scribblez/board.h"
#include "scribblez/game.h"
#include "scribblez/tile_counts.h"

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

struct ParsedGcgGame {
  std::array<std::string, 2> player_names = {"Player 1", "Player 2"};
  std::vector<ParsedGcgTurn> turns;
  std::vector<ParsedGcgSnapshot> snapshots;
  GameLogStorage game_log;

  GameLogStorage to_game_log_storage() const;
};

bool read_gcg_text(const std::string& gcg_text, ParsedGcgGame* out_game,
                   std::string* error_message);

}  // namespace scribblez
