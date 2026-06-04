#pragma once

#include <array>
#include <cstdint>

namespace scribblez {

// A single Scrabble tile, stored in one byte. A tile is a letter A..Z, a blank,
// or -- when it is the content of a board square -- the empty sentinel.
//
// A Tile converts implicitly to its underlying code (0..25 = A..Z, 26 = blank,
// 27 = empty) so it can be used directly as an array index or in arithmetic.
// Construct with the factories or the BLANK / EMPTY_SQUARE constants below.
class Tile {
 public:
  constexpr Tile() = default;  // empty square

  static constexpr Tile of(int letter_index) { return Tile(static_cast<uint8_t>(letter_index)); }
  static constexpr Tile blank() { return Tile(kBlank); }
  static constexpr Tile empty() { return Tile(kEmpty); }
  static constexpr Tile from_char(char c);

  constexpr operator uint8_t() const { return code_; }  // usable as an index
  constexpr uint8_t index() const { return code_; }
  constexpr bool is_blank() const { return code_ == kBlank; }
  constexpr bool is_empty() const { return code_ == kEmpty; }
  constexpr char to_char() const;
  int value() const;  // Scrabble points; a blank or empty scores 0.

  constexpr Tile& operator++() {
    ++code_;
    return *this;
  }

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

// Standard English Scrabble point values, indexed by letter (0..25).
extern const std::array<int, 26> TILE_VALUES;

// Standard English tile counts. Index 0..25 = A..Z, index 26 = blank.
extern const std::array<int, 27> TILE_COUNTS;

constexpr char Tile::to_char() const {
  if (code_ == kBlank) return '?';
  if (code_ == kEmpty) return '.';
  return static_cast<char>('A' + code_);
}

constexpr Tile Tile::from_char(char c) {
  if (c == '?' || c == '_') return blank();
  if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  return of(c - 'A');
}

inline int Tile::value() const { return (is_blank() || is_empty()) ? 0 : TILE_VALUES[code_]; }

}  // namespace scribblez
