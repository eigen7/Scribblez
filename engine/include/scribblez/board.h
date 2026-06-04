#pragma once

#include "scribblez/tile.h"

#include <array>
#include <string>
#include <vector>

namespace scribblez {

constexpr int BOARD_SIZE = 15;
constexpr int CENTER = 7;

enum class Premium : uint8_t { NONE = 0, DLS, TLS, DWS, TWS };

struct Square {
  Tile letter = EMPTY_SQUARE;  // 0..25 if occupied, EMPTY_SQUARE otherwise
  bool is_blank = false;       // true if the placed tile was originally a blank
};

inline bool is_empty(Square s) { return s.letter == EMPTY_SQUARE; }

struct PlacedTile {
  int row;
  int col;
  Tile letter;    // 0..25 (the letter shown on the board)
  bool is_blank;  // true iff this tile came from a blank in the rack
};

class Board {
 public:
  Board();

  Square at(int r, int c) const { return squares_[r * BOARD_SIZE + c]; }
  void set(int r, int c, Square s) { squares_[r * BOARD_SIZE + c] = s; }
  bool in_bounds(int r, int c) const {
    return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
  }
  bool empty_board() const;

  Premium premium_at(int r, int c) const { return PREMIUM[r * BOARD_SIZE + c]; }

  // Apply a list of placed tiles to the board.
  void apply(const std::vector<PlacedTile>& tiles);

  // Pretty-print the board to a string.
  std::string to_string() const;

  static const std::array<Premium, BOARD_SIZE * BOARD_SIZE> PREMIUM;

 private:
  std::array<Square, BOARD_SIZE * BOARD_SIZE> squares_{};
};

}  // namespace scribblez
