#include "game/glyph.h"

namespace scribblez {

constexpr Glyph Glyph::of_blank(Tile letter) {
  return Glyph(static_cast<uint8_t>(letter.index() + 27));
}

// A tile played as a designated blank renders that letter but scores nothing.
constexpr Glyph Glyph::played(Tile letter, bool is_blank) {
  return is_blank ? of_blank(letter) : of(letter);
}

constexpr Tile Glyph::letter() const {  // valid iff has_letter()
  return Tile::of(code_ <= 26 ? code_ - 1 : code_ - 27);
}

constexpr bool Glyph::is_vowel() const {
  // A designated blank that shows a vowel (codes 27..52) counts as a vowel too,
  // since has_letter() spans both played letters and designated blanks.
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
  return static_cast<char>('A' + (code_ <= 26 ? code_ - 1 : code_ - 27));
}

}  // namespace scribblez
