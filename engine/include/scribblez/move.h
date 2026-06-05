#pragma once

#include "scribblez/board.h"
#include "scribblez/glyph.h"
#include "scribblez/tile.h"

#include <array>
#include <cstdint>
#include <string>

namespace scribblez {

enum class MoveType : uint8_t { PLAY, EXCHANGE, PASS };

// A move in compact, fixed-size form (16 bytes). The full word and tile
// positions are not stored -- they are reconstructed from the board.
//
// PLAY:     `glyphs` are the newly placed tiles in order along the main word;
//           start_row/col is the word's first square and `horizontal` its
//           direction. Squares between/after that are read from the board.
// EXCHANGE: `glyphs` are the tiles surrendered (an unassigned blank is
//           Glyph::blank()).
// PASS:     `glyphs` is empty.
// Unused trailing slots are empty glyphs.
struct Move {
  MoveType type = MoveType::PASS;
  bool horizontal = true;
  int8_t start_row = 0;
  int8_t start_col = 0;
  std::array<Glyph, RACK_SIZE> glyphs{};
  int score = 0;

  // Number of leading non-empty glyphs (placed/surrendered tiles).
  int num_glyphs() const;

  // Reconstruct the move's main word (full word, uppercase) from the board
  // as it stood *before* this move was applied.
  std::string main_word(const Board& board) const;
};

static_assert(sizeof(Move) == 16, "Move should pack into 16 bytes");

}  // namespace scribblez

#include "inlines/scribblez/move.inl"
