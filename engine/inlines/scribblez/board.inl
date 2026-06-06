#include "scribblez/board.h"

namespace scribblez {

constexpr char Premium::display_char() const {
  if (kind_ == kDLS) return 'd';
  if (kind_ == kTLS) return 't';
  if (kind_ == kDWS) return 'D';
  if (kind_ == kTWS) return 'T';
  return '.';
}

constexpr const char* Premium::code() const {
  if (kind_ == kDLS) return "DL";
  if (kind_ == kTLS) return "TL";
  if (kind_ == kDWS) return "DW";
  if (kind_ == kTWS) return "TW";
  return nullptr;
}

inline bool Board::in_bounds(int r, int c) const {
  return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

}  // namespace scribblez
