#include "scribblez/board.h"

namespace scribblez {

inline bool Board::in_bounds(int r, int c) const {
  return r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE;
}

}  // namespace scribblez
