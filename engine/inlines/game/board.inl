#include "game/board.h"

namespace scribblez {

constexpr char Premium::display_char() const {
  if (kind_ == kDLS) return '\'';
  if (kind_ == kTLS) return '"';
  if (kind_ == kDWS) return '-';
  if (kind_ == kTWS) return '=';
  return ' ';
}

inline Premium decode(char c) {
  switch (c) {
    case '\'':
      return Premium::DLS;
    case '"':
      return Premium::TLS;
    case '-':
      return Premium::DWS;
    case '=':
      return Premium::TWS;
    default:
      return Premium::NONE;
  }
}

constexpr const char* Premium::code() const {
  if (kind_ == kDLS) return "DL";
  if (kind_ == kTLS) return "TL";
  if (kind_ == kDWS) return "DW";
  if (kind_ == kTWS) return "TW";
  return nullptr;
}

inline void Board::set(int r, int c, Glyph g) {
  squares_[r * BOARD_SIZE + c] = g;
  caches_valid_ = false;
}

inline bool Board::in_bounds(int r, int c) const {
  return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

inline Glyph Board::oriented_at(int r, int c, bool transposed) const {
  return transposed ? at(c, r) : at(r, c);
}

}  // namespace scribblez
