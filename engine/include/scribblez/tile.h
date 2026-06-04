#pragma once
#include <array>
#include <cstdint>

namespace scribblez {

// 0..25 = letters A..Z. 26 = blank tile (in a rack/bag). 27 = empty board square.
using Letter = uint8_t;
constexpr Letter BLANK = 26;
constexpr Letter EMPTY_SQUARE = 27;

inline char letter_to_char(Letter l) {
  if (l == BLANK) return '?';
  if (l == EMPTY_SQUARE) return '.';
  return static_cast<char>('A' + l);
}

inline Letter char_to_letter(char c) {
  if (c == '?' || c == '_') return BLANK;
  if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  return static_cast<Letter>(c - 'A');
}

// Standard English Scrabble point values, indexed by Letter (0..25).
extern const std::array<int, 26> LETTER_VALUES;

// Standard English tile counts. Index 0..25 = A..Z, index 26 = blank.
extern const std::array<int, 27> TILE_COUNTS;

constexpr int RACK_SIZE = 7;

}  // namespace scribblez
