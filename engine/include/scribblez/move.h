#pragma once

#include "scribblez/board.h"
#include "scribblez/glyph.h"
#include "scribblez/tile.h"
#include "scribblez/tile_counts.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace scribblez {

enum class MoveType : uint8_t { PLAY, EXCHANGE, PASS };

// A move in compact, fixed-size form (16 bytes). The full word and tile
// positions are not stored -- they are reconstructed from the board.
//
// `glyphs` holds only the played/surrendered tiles: slots [0, num_glyphs())
// (in word order for a PLAY); the remaining slots are empty. A move records no
// leave: the leave is a property of the rack a move is generated against, not
// of the move itself, so the equity layer derives it on demand.
//
// PLAY:     the played glyphs are the newly placed tiles in order along the
//           main word. `start()` is the word's cross-axis coordinate (its row
//           for a horizontal play, its column for a vertical one) and
//           `square_mask()` is an absolute bitmask over the play's lane: bit k
//           is set iff lane cell k holds a newly placed tile (k is a column for
//           a horizontal play, a row for a vertical one). Squares in the run
//           that are not set in the mask are read from the board.
//
//           Example: an `E` already sits at (row 7, col 8), and READ is played
//           horizontally across row 7 -- R, A, D newly placed at cols 7, 9, 10,
//           reusing the board's E at col 8. Then start() == 7, square_mask has
//           bits 7, 9, 10 set (bit 8 clear -> read from the board), and the
//           stored glyphs are {R, A, D} in word order (the E is not stored).
//           main_word() interleaves stored glyphs with board tiles to recover
//           "READ".
// EXCHANGE: the played glyphs are the tiles surrendered (an unassigned blank is
//           Glyph::blank()). `square_mask()` is unused (0).
// PASS:     no played glyphs. `square_mask()` is unused (0).
//
// `score()` is an unsigned 16-bit count; the theoretical Scrabble maximum for a
// single play is well under 2^16.
class Move {
 public:
  Move() = default;  // a PASS

  // Factories -- the only places that populate Move's private representation.
  //
  // A PLAY: `played[0, num_played)` are the placed glyphs in word order (their
  // count equals popcount(square_mask)).
  static Move play(bool horizontal, int start, uint16_t square_mask, uint16_t score,
                   const Glyph* played, int num_played);
  // An EXCHANGE surrendering `tiles` (laid out sorted).
  static Move exchange(const TileCounts& tiles);
  // A PASS.
  static Move pass() { return Move{}; }

  MoveType type() const { return type_; }
  bool horizontal() const { return horizontal_; }
  int start() const { return start_; }
  uint16_t square_mask() const { return square_mask_; }
  uint16_t score() const { return score_; }

  // Number of played (PLAY) or surrendered (EXCHANGE) tiles; 0 for a PASS.
  int num_glyphs() const { return num_played_; }

  // The i-th played/surrendered glyph (0 <= i < num_glyphs()).
  Glyph glyph(int i) const { return glyphs_[i]; }

  // (row, col) of the main word's first square, recovered by walking back
  // through existing tiles on `board` (the board as it stood before the move).
  std::pair<int, int> word_origin(const Board& board) const;

  // Reconstruct the move's main word (full word, uppercase) from the board
  // as it stood *before* this move was applied.
  std::string main_word(const Board& board) const;

 private:
  MoveType type_ = MoveType::PASS;         // 1 B
  bool horizontal_ = true;                 // 1 B
  int8_t start_ = 0;                       // 1 B; cross-axis coord (PLAY only)
  uint8_t num_played_ = 0;                 // 1 B; played/surrendered tile count
  std::array<Glyph, RACK_SIZE> glyphs_{};  // 7 B; played/surrendered tiles only
  uint8_t reserved_ = 0;                   // 1 B; explicit (zeroed) alignment padding, so
                                           // serialized Moves (.slog / .sobs) are
                                           // byte-deterministic
  uint16_t square_mask_ = 0;               // 2 B; PLAY only; see class comment
  uint16_t score_ = 0;                     // 2 B
};

static_assert(sizeof(Move) == 16, "Move should pack into 16 bytes");

// Calls f(row, col) once per board square `m` places a tile on, in lane order.
// EXCHANGE and PASS place nothing, so f is never called for them.
template <typename F>
void visit_placed_squares(const Move& m, F&& f);

}  // namespace scribblez

#include "inlines/scribblez/move.inl"
