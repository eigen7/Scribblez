#pragma once

#include "scribblez/glyph.h"
#include "scribblez/tile.h"

#include <array>
#include <string>

namespace scribblez {

constexpr int BOARD_SIZE = 15;
constexpr int CENTER = 7;

enum class Premium : uint8_t { NONE = 0, DLS, TLS, DWS, TWS };

struct Move;  // forward declaration

class Board {
 public:
  Board();

  Glyph at(int r, int c) const { return squares_[r * BOARD_SIZE + c]; }
  void set(int r, int c, Glyph g) { squares_[r * BOARD_SIZE + c] = g; }
  bool in_bounds(int r, int c) const;
  bool empty_board() const;

  Premium premium_at(int r, int c) const { return PREMIUM[r * BOARD_SIZE + c]; }

  // Place the move's new tiles on the board.
  void apply(const Move& move);

  // Pretty-print the board to a string.
  std::string to_string() const;

  static const std::array<Premium, BOARD_SIZE * BOARD_SIZE> PREMIUM;

 private:
  std::array<Glyph, BOARD_SIZE * BOARD_SIZE> squares_{};
};

}  // namespace scribblez

#include "inlines/scribblez/board.inl"
