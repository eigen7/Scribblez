#include "scribblez/board.h"

#include <array>
#include <string>

namespace scribblez {

namespace {

// Encoded as: '.' NONE, 'd' DLS, 't' TLS, 'D' DWS, 'T' TWS.
// Center (7,7) is treated as DWS for first-move scoring.
// clang-format off
constexpr const char* kPremiumLayout[BOARD_SIZE] = {
    "T..d...T...d..T",
    ".D...t...t...D.",
    "..D...d.d...D..",
    "d..D...d...D..d",
    "....D.....D....",
    ".t...t...t...t.",
    "..d...d.d...d..",
    "T..d...D...d..T",
    "..d...d.d...d..",
    ".t...t...t...t.",
    "....D.....D....",
    "d..D...d...D..d",
    "..D...d.d...D..",
    ".D...t...t...D.",
    "T..d...T...d..T",
};
// clang-format on

Premium decode(char c) {
  switch (c) {
    case 'd':
      return Premium::DLS;
    case 't':
      return Premium::TLS;
    case 'D':
      return Premium::DWS;
    case 'T':
      return Premium::TWS;
    default:
      return Premium::NONE;
  }
}

std::array<Premium, BOARD_SIZE * BOARD_SIZE> build_premium() {
  std::array<Premium, BOARD_SIZE * BOARD_SIZE> out{};
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      out[r * BOARD_SIZE + c] = decode(kPremiumLayout[r][c]);
    }
  }
  return out;
}

}  // namespace

const std::array<Premium, BOARD_SIZE * BOARD_SIZE> Board::PREMIUM = build_premium();

Board::Board() {
  for (auto& s : squares_) s = Glyph::empty();
}

bool Board::empty_board() const {
  for (auto& s : squares_)
    if (!s.is_empty()) return false;
  return true;
}

void Board::apply(const std::vector<PlacedTile>& tiles) {
  for (const auto& t : tiles) {
    set(t.row, t.col, t.glyph);
  }
}

std::string Board::to_string() const {
  std::string s;
  s.reserve((BOARD_SIZE + 1) * (BOARD_SIZE + 4));
  s += "   ";
  for (int c = 0; c < BOARD_SIZE; ++c) {
    s.push_back(static_cast<char>('A' + c));
    s.push_back(' ');
  }
  s.push_back('\n');
  for (int r = 0; r < BOARD_SIZE; ++r) {
    char buf[4];
    std::snprintf(buf, sizeof(buf), "%2d ", r + 1);
    s += buf;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph sq = at(r, c);
      if (sq.is_empty()) {
        switch (premium_at(r, c)) {
          case Premium::NONE:
            s += ". ";
            break;
          case Premium::DLS:
            s += "d ";
            break;
          case Premium::TLS:
            s += "t ";
            break;
          case Premium::DWS:
            s += "D ";
            break;
          case Premium::TWS:
            s += "T ";
            break;
        }
      } else {
        s.push_back(sq.is_blank() ? static_cast<char>('a' + sq.letter())
                                  : static_cast<char>('A' + sq.letter()));
        s.push_back(' ');
      }
    }
    s.push_back('\n');
  }
  return s;
}

}  // namespace scribblez
