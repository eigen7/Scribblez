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

// A move in compact, fixed-size form (16 bytes). Neither the full word nor the
// tile positions are stored -- both are reconstructed from the board. Nor is
// the leave, which belongs to the rack a move was generated against rather than
// to the move, so the equity layer derives it on demand.
//
// PLAY:     `glyphs` holds the newly placed tiles in order along the main word.
//           `start()` is the word's cross-axis coordinate and `square_mask()`
//           an absolute bitmask over its lane: bit k is set iff lane cell k
//           holds a newly placed tile. Cells of the run that are clear in the
//           mask are read from the board.
//
//           Example: an `E` already sits at (row 7, col 8), and READ is played
//           horizontally across row 7 -- R, A, D newly placed at cols 7, 9, 10.
//           Then start() == 7, square_mask has bits 7, 9, 10 set, and the
//           stored glyphs are {R, A, D}; main_word() interleaves them with the
//           board's E to recover "READ".
// EXCHANGE: `glyphs` holds the surrendered tiles (an unassigned blank is
//           Glyph::blank()); square_mask() is unused.
// PASS:     no glyphs; square_mask() is unused.
class Move {
 public:
  Move() = default;  // a PASS

  // The factories are the only places that populate the representation.
  //
  // `played[0, num_played)` are the placed glyphs in word order, so their count
  // equals popcount(square_mask).
  static Move play(bool horizontal, int start, uint16_t square_mask, uint16_t score,
                   const Glyph* played, int num_played);
  static Move exchange(const TileCounts& tiles);
  static Move pass() { return Move{}; }

  MoveType type() const { return type_; }
  bool horizontal() const { return horizontal_; }
  int start() const { return start_; }
  uint16_t square_mask() const { return square_mask_; }
  uint16_t score() const { return score_; }

  int num_glyphs() const { return num_played_; }

  Glyph glyph(int i) const { return glyphs_[i]; }

  // `board` must be as it stood BEFORE this move was applied.
  std::pair<int, int> word_origin(const Board& board) const;
  std::string main_word(const Board& board) const;

 private:
  MoveType type_ = MoveType::PASS;         // 1 B
  bool horizontal_ = true;                 // 1 B
  int8_t start_ = 0;                       // 1 B; cross-axis coord (PLAY only)
  uint8_t num_played_ = 0;                 // 1 B
  std::array<Glyph, RACK_SIZE> glyphs_{};  // 7 B
  uint8_t reserved_ = 0;                   // 1 B; zeroed padding, so serialized Moves
                                           // (.slog / .sobs) are byte-deterministic
  uint16_t square_mask_ = 0;               // 2 B; PLAY only; see class comment
  uint16_t score_ = 0;                     // 2 B; a play's max is well under 2^16
};

static_assert(sizeof(Move) == 16, "Move should pack into 16 bytes");

// Byte-wise equality is exact: every Move byte is meaningful or explicitly
// zeroed (reserved_), and the factories produce canonical orderings (play
// glyphs in lane order, exchange tiles sorted).
inline bool operator==(const Move& a, const Move& b) { return std::memcmp(&a, &b, 16) == 0; }
inline bool operator!=(const Move& a, const Move& b) { return !(a == b); }

// Calls f(row, col) once per board square `m` places a tile on, in lane order.
template <typename F>
void visit_placed_squares(const Move& m, F&& f);

}  // namespace scribblez

#include "inlines/game/move.inl"
