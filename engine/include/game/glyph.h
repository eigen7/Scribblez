#pragma once

#include "game/tile.h"

#include <cstdint>

namespace scribblez {

// The face of a board square (or a slot in a move): empty, a played letter
// (optionally a designated blank), or an unassigned blank (a blank with no
// chosen letter -- only meaningful in a move/rack, never on the board).
//
// One byte:
//   0        empty
//   1..26    played letter A..Z            (letter index = code - 1)
//   27..52   designated blank, shows A..Z  (letter index = code - 27)
//   53       unassigned blank
class Glyph {
 public:
  constexpr Glyph() = default;  // empty, so a zero-filled array is all-empty

  static constexpr Glyph empty() { return Glyph(0); }
  static constexpr Glyph of(Tile letter) { return Glyph(static_cast<uint8_t>(letter.index() + 1)); }
  static constexpr Glyph of_blank(Tile letter);
  static constexpr Glyph blank() { return Glyph(53); }

  static constexpr Glyph played(Tile letter, bool is_blank);

  constexpr bool is_empty() const { return code_ == 0; }
  constexpr bool is_blank() const { return code_ >= 27; }  // designated or not
  constexpr bool has_letter() const { return code_ >= 1 && code_ <= 52; }
  constexpr Tile letter() const;    // valid iff has_letter()
  constexpr bool is_vowel() const;  // false unless has_letter()
  int value() const { return has_letter() && !is_blank() ? letter().value() : 0; }
  constexpr char to_char() const;
  constexpr uint8_t code() const { return code_; }

  constexpr Tile rack_tile() const { return is_blank() ? BLANK : letter(); }

  static constexpr Glyph exchanging(Tile t) { return t.is_blank() ? blank() : of(t); }

  constexpr bool operator==(Glyph o) const { return code_ == o.code_; }
  constexpr bool operator!=(Glyph o) const { return code_ != o.code_; }

 private:
  explicit constexpr Glyph(uint8_t code) : code_(code) {}
  uint8_t code_ = 0;
};

static_assert(sizeof(Glyph) == 1, "Glyph must pack into one byte");

}  // namespace scribblez

#include "inlines/game/glyph.inl"
