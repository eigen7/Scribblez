#include "scribblez/tile.h"

namespace scribblez {

constexpr Tile& Tile::operator++() {
  ++code_;
  return *this;
}

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
