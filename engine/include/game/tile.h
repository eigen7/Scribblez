#pragma once

#include <array>
#include <cstdint>

namespace scribblez {

// A tile in one byte: a letter A..Z, a blank, or the empty sentinel. Converts
// implicitly to its code (0..25 = A..Z, 26 = blank, 27 = empty) for use as an
// array index.
class Tile {
 public:
  constexpr Tile() = default;  // empty square

  static constexpr Tile of(int letter_index) { return Tile(uint8_t(letter_index)); }
  static constexpr Tile blank() { return Tile(kBlank); }
  static constexpr Tile empty() { return Tile(kEmpty); }
  static constexpr Tile from_char(char c);  // either case; '?' or '_' is a blank
  // A..Z in either case as its letter; empty for any other character.
  static constexpr Tile letter_from_char(char c);

  constexpr operator uint8_t() const { return code_; }
  constexpr uint8_t index() const { return code_; }
  constexpr bool is_blank() const { return code_ == kBlank; }
  constexpr bool is_empty() const { return code_ == kEmpty; }
  constexpr char to_char() const;
  int value() const;  // Scrabble points; a blank or empty scores 0.

  constexpr Tile& operator++();

 private:
  static constexpr uint8_t kBlank = 26;
  static constexpr uint8_t kEmpty = 27;
  explicit constexpr Tile(uint8_t code) : code_(code) {}
  uint8_t code_ = kEmpty;
};

static_assert(sizeof(Tile) == 1, "Tile must pack into one byte");

inline constexpr Tile BLANK = Tile::blank();
inline constexpr Tile EMPTY_SQUARE = Tile::empty();

constexpr int RACK_SIZE = 7;

// The kinds of tile a player can hold: the letters A..Z and the blank, i.e. the
// Tile codes below the empty sentinel.
constexpr int TILE_KINDS = 27;

// Standard English point values and tile distribution, indexed by Tile code.
extern const std::array<int, 26> TILE_VALUES;
extern const std::array<int, TILE_KINDS> TILE_COUNTS;

}  // namespace scribblez

#include "inlines/game/tile.inl"
