#include "game/glyph.h"

namespace scribblez {

constexpr Glyph Glyph::of_blank(Tile letter) { return Glyph(uint8_t(letter.index() + 27)); }

constexpr Glyph Glyph::played(Tile letter, bool is_blank) {
  return is_blank ? of_blank(letter) : of(letter);
}

constexpr Tile Glyph::letter() const { return Tile::of(code_ <= 26 ? code_ - 1 : code_ - 27); }

constexpr bool Glyph::is_vowel() const {
  if (!has_letter()) return false;
  switch (letter().index()) {
    case 0:   // A
    case 4:   // E
    case 8:   // I
    case 14:  // O
    case 20:  // U
      return true;
    default:
      return false;
  }
}

constexpr char Glyph::to_char() const {
  if (is_empty()) return '.';
  if (code_ == 53) return '?';
  return 'A' + (code_ <= 26 ? code_ - 1 : code_ - 27);
}

}  // namespace scribblez
