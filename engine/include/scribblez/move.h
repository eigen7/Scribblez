#pragma once
#include <string>
#include <vector>

#include "scribblez/board.h"
#include "scribblez/tile.h"

namespace scribblez {

enum class MoveType { PLAY, EXCHANGE, PASS };

struct Move {
  MoveType type = MoveType::PASS;

  // For PLAY:
  std::vector<PlacedTile> tiles;     // tiles newly placed, ordered along main word
  bool horizontal = true;            // direction of main word
  int start_row = 0;                 // first square of main word (may be an existing tile)
  int start_col = 0;
  std::string main_word;             // full main word (including tiles already on board)
  int score = 0;                     // total score of the play

  // For EXCHANGE:
  std::vector<Letter> exchanged;     // tiles taken from rack and returned to bag
};

}  // namespace scribblez
