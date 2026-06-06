#pragma once

#include "scribblez/glyph.h"

#include <array>
#include <string>

namespace scribblez {

constexpr int BOARD_SIZE = 15;
constexpr int CENTER = 7;

class Premium {
 public:
  enum Kind : uint8_t { kNone = 0, kDLS, kTLS, kDWS, kTWS };

  constexpr Premium() : kind_(kNone) {}
  constexpr explicit Premium(Kind k) : kind_(k) {}

  static const Premium NONE;
  static const Premium DLS;
  static const Premium TLS;
  static const Premium DWS;
  static const Premium TWS;

  constexpr bool operator==(Premium o) const { return kind_ == o.kind_; }
  constexpr bool operator!=(Premium o) const { return kind_ != o.kind_; }

  constexpr int letter_mult() const { return kind_ == kDLS ? 2 : kind_ == kTLS ? 3 : 1; }
  constexpr int word_mult()   const { return kind_ == kDWS ? 2 : kind_ == kTWS ? 3 : 1; }

  constexpr char display_char() const;

  // Returns "DL"/"TL"/"DW"/"TW" for premium squares, nullptr for NONE.
  constexpr const char* code() const;

 private:
  Kind kind_;
};
static_assert(sizeof(Premium) == 1);

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
