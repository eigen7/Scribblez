#pragma once

#include "scribblez/glyph.h"

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

constexpr char Glyph::to_char() const {
  if (is_empty()) return '.';
  if (code_ == 53) return '?';
  return static_cast<char>('A' + (code_ <= 26 ? code_ - 1 : code_ - 27));
}

}  // namespace scribblez
