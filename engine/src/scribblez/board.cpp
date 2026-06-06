#include "scribblez/board.h"

#include "scribblez/move.h"

#include <array>
#include <string>

namespace scribblez {

const Premium Premium::NONE = Premium(Premium::kNone);
const Premium Premium::DLS  = Premium(Premium::kDLS);
const Premium Premium::TLS  = Premium(Premium::kTLS);
const Premium Premium::DWS  = Premium(Premium::kDWS);
const Premium Premium::TWS  = Premium(Premium::kTWS);
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

void Board::apply(const Move& move) {
  if (move.type != MoveType::PLAY) return;
  const int dr = move.horizontal ? 0 : 1;
  const int dc = move.horizontal ? 1 : 0;
  int r = move.start_row, c = move.start_col;
  const int n = move.num_glyphs();
  for (int gi = 0; gi < n; ++gi) {
    while (in_bounds(r, c) && !at(r, c).is_empty()) {
      r += dr;
      c += dc;
    }
    if (!in_bounds(r, c)) break;
    set(r, c, move.glyphs[gi]);
    r += dr;
    c += dc;
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
        s.push_back(premium_at(r, c).display_char());
        s.push_back(' ');
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
