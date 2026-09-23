#pragma once

#include "game/board.h"
#include "game/glyph.h"
#include "game/tile.h"
#include "game/tile_counts.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

namespace scribblez {

enum class MoveType : uint8_t { PLAY, EXCHANGE, PASS };

// A move in compact, fixed-size form (16 bytes). The full word is not stored;
// it is reconstructed from the board. Nor is the leave, which depends on the
// rack rather than the move.
//
// PLAY:     `glyphs` holds the newly placed tiles in order along the main word.
//           `start()` is the lane (the row of a horizontal play, the column of
//           a vertical one) and `square_mask()` marks the newly placed cells:
//           bit k is set iff lane cell k gets a tile. The word's other cells
//           are read from the board.
//
//           Example: an `E` already sits at (row 7, col 8), and READ is played
//           horizontally across row 7 -- R, A, D newly placed at cols 7, 9, 10.
//           Then start() == 7, square_mask has bits 7, 9, 10 set, and the
//           stored glyphs are {R, A, D}; main_word() interleaves them with the
//           board's E to recover "READ".
// EXCHANGE: `glyphs` holds the surrendered tiles (an unassigned blank is
//           Glyph::blank()); square_mask() is unused.
// PASS:     no glyphs; square_mask() is unused.
//
// Every move carries a frame bit, like Board::transposed(): the game's natural
// frame (false) or its diagonal transpose (true). A Move is only meaningful
// against a Board of the same frame, which debug builds assert.
class Move {
 public:
  Move() = default;  // a PASS

  // `played[0, num_played)` are the placed glyphs in word order, one per set
  // bit of `square_mask`. `transposed` is the frame of the board the play was
  // built against.
  static Move play(bool horizontal, int start, uint16_t square_mask, uint16_t score,
                   const Glyph* played, int num_played, bool transposed = false);
  static Move exchange(const TileCounts& tiles);
  static Move pass() { return Move{}; }

  MoveType type() const { return type_; }
  bool horizontal() const { return horizontal_; }
  int start() const { return start_; }
  uint16_t square_mask() const { return square_mask_; }
  uint16_t score() const { return score_; }
  bool transposed() const { return transposed_; }

  // This move in the other frame. A PLAY's direction flips while `start` and
  // `square_mask` are unchanged.
  Move transpose() const;

  int num_glyphs() const { return num_played_; }

  Glyph glyph(int i) const { return glyphs_[i]; }

  // `board` must be as it stood BEFORE this move was applied.
  std::pair<int, int> word_origin(const Board& board) const;
  std::string main_word(const Board& board) const;

 private:
  MoveType type_ = MoveType::PASS;         // 1 B
  bool horizontal_ = true;                 // 1 B
  int8_t start_ = 0;                       // 1 B; lane index (PLAY only)
  uint8_t num_played_ = 0;                 // 1 B
  std::array<Glyph, RACK_SIZE> glyphs_{};  // 7 B
  bool transposed_ = false;                // 1 B; frame bit; 0 in .slog/.mset/.sobs files
  uint16_t square_mask_ = 0;               // 2 B; PLAY only; see class comment
  uint16_t score_ = 0;                     // 2 B
};

static_assert(sizeof(Move) == 16, "Move should pack into 16 bytes");

// Byte-wise equality is exact: the struct has no padding, unused glyph slots
// stay zero, and the factories order glyphs canonically (plays in lane order,
// exchanges sorted). A move and its transpose compare unequal.
inline bool operator==(const Move& a, const Move& b) { return std::memcmp(&a, &b, 16) == 0; }
inline bool operator!=(const Move& a, const Move& b) { return !(a == b); }

// Calls f(row, col) once per board square `m` places a tile on, in lane order.
template <typename F>
void visit_placed_squares(const Move& m, F&& f);

}  // namespace scribblez

#include "inlines/game/move.inl"
