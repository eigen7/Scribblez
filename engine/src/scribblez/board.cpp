#include "scribblez/board.h"

#include "scribblez/dictionary.h"
#include "scribblez/move.h"
#include "scribblez/tile.h"

#include <array>
#include <string>
#include <utility>

namespace scribblez {

const Premium Premium::NONE = Premium(Premium::kNone);
const Premium Premium::DLS = Premium(Premium::kDLS);
const Premium Premium::TLS = Premium(Premium::kTLS);
const Premium Premium::DWS = Premium(Premium::kDWS);
const Premium Premium::TWS = Premium(Premium::kTWS);
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
  const bool had_caches = caches_valid_;
  const bool was_empty = empty_board();
  const int dr = move.horizontal ? 0 : 1;
  const int dc = move.horizontal ? 1 : 0;
  int r = move.start_row, c = move.start_col;
  const int n = move.num_glyphs();
  std::array<std::pair<int, int>, RACK_SIZE> placed{};
  int np = 0;
  for (int gi = 0; gi < n; ++gi) {
    while (in_bounds(r, c) && !at(r, c).is_empty()) {
      r += dr;
      c += dc;
    }
    if (!in_bounds(r, c)) break;
    set(r, c, move.glyphs[gi]);  // clears caches_valid_
    placed[np++] = {r, c};
    r += dr;
    c += dc;
  }
  if (np == 0 || !had_caches) return;  // nothing placed, or caches were stale anyway

  // Keep the caches in sync without a full rescan. The first move (empty board
  // becoming non-empty) flips the anchor model, so just rebuild once.
  if (was_empty) {
    recompute_all_caches();
  } else {
    update_caches_after_place(placed.data(), np);
  }
  caches_valid_ = true;
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

// ---------------------------------------------------------------------------
// Persistent move-generation caches (cross-checks + GADDAG anchors).
//
// These mirror Macondo's board-resident cross-sets and anchors: computed once
// for a position and then updated incrementally as tiles are placed, so the
// move generator never rescans the whole board on a turn that did not change it
// (e.g. PASS/EXCHANGE) and only touches the affected squares on a PLAY.
// ---------------------------------------------------------------------------

CrossCheck Board::cross_check_at(bool t, int r, int c) const {
  CrossCheck cc;
  if (!oriented_at(r, c, t).is_empty()) return cc;  // filled squares: unused default

  // Maximal existing perpendicular run touching (r, c): rows [top, r-1] above
  // and [r+1, bot] below at fixed column c (in the current orientation).
  int top = r - 1;
  while (top >= 0 && !oriented_at(top, c, t).is_empty()) --top;
  ++top;
  int bot = r + 1;
  while (bot < BOARD_SIZE && !oriented_at(bot, c, t).is_empty()) ++bot;
  --bot;

  cc.has_neighbor = (top < r) || (bot > r);
  if (!cc.has_neighbor) {
    cc.mask = kAllLettersMask;
    cc.score = 0;
    return cc;
  }

  const Dictionary& dict = *dict_;
  uint32_t prefix_node = dict.root();
  int prefix_score = 0;
  bool prefix_ok = true;
  for (int rr = top; rr <= r - 1; ++rr) {
    Glyph sq = oriented_at(rr, c, t);
    auto tr = dict.step(prefix_node, sq.letter());
    if (!tr.valid) {
      prefix_ok = false;
      break;
    }
    prefix_node = tr.next;
    if (!sq.is_blank()) prefix_score += TILE_VALUES[sq.letter()];
  }
  int suffix_score = 0;
  for (int rr = r + 1; rr <= bot; ++rr) {
    Glyph sq = oriented_at(rr, c, t);
    if (!sq.is_blank()) suffix_score += TILE_VALUES[sq.letter()];
  }
  cc.score = prefix_score + suffix_score;

  uint32_t mask = 0;
  if (prefix_ok) {
    for (Tile L = Tile::of(0); L < 26; ++L) {
      auto tr_l = dict.step(prefix_node, L);
      if (!tr_l.valid) continue;
      bool acc = tr_l.accepts;
      uint32_t node = tr_l.next;
      bool ok = true;
      for (int rr = r + 1; rr <= bot; ++rr) {
        auto tr_s = dict.step(node, oriented_at(rr, c, t).letter());
        if (!tr_s.valid) {
          ok = false;
          break;
        }
        acc = tr_s.accepts;
        node = tr_s.next;
      }
      if (ok && acc) mask |= (1u << L);
    }
  }
  cc.mask = mask;
  return cc;
}

bool Board::gaddag_anchor_at(bool t, int r, int c) const {
  // GADDAG anchors (direction-specific, along increasing column in this view):
  //   - an occupied square is an anchor iff nothing is immediately to its right;
  //   - an empty square is an anchor iff both horizontal neighbors are empty and
  //     it has a tile directly above or below (a pure cross-hook).
  const bool here = !oriented_at(r, c, t).is_empty();
  const bool tile_left = c > 0 && !oriented_at(r, c - 1, t).is_empty();
  const bool tile_right = c < BOARD_SIZE - 1 && !oriented_at(r, c + 1, t).is_empty();
  const bool tile_above = r > 0 && !oriented_at(r - 1, c, t).is_empty();
  const bool tile_below = r < BOARD_SIZE - 1 && !oriented_at(r + 1, c, t).is_empty();
  if (here) return !tile_right;
  return !tile_left && !tile_right && (tile_above || tile_below);
}

void Board::recompute_all_caches() const {
  const bool empty = empty_board();
  for (int t = 0; t < 2; ++t) {
    auto& cross = cross_[t];
    auto& anchor = ganchor_[t];
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int c = 0; c < BOARD_SIZE; ++c) cross[r * BOARD_SIZE + c] = cross_check_at(t, r, c);
    if (empty) {
      anchor.fill(false);
      anchor[CENTER * BOARD_SIZE + CENTER] = true;  // sole opening anchor
    } else {
      for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c) anchor[r * BOARD_SIZE + c] = gaddag_anchor_at(t, r, c);
    }
  }
}

void Board::update_caches_after_place(const std::pair<int, int>* placed, int n) const {
  for (int t = 0; t < 2; ++t) {
    auto& cross = cross_[t];
    for (int i = 0; i < n; ++i) {
      // View coordinates of the placed square in this orientation.
      const int vr = t ? placed[i].second : placed[i].first;
      const int vc = t ? placed[i].first : placed[i].second;
      // The placed square is now filled; its cross-check is unused.
      cross[vr * BOARD_SIZE + vc] = CrossCheck{};
      // Only the empty squares at the two ends of the (now extended)
      // perpendicular run through column vc can change.
      int top = vr;
      while (top - 1 >= 0 && !oriented_at(top - 1, vc, t).is_empty()) --top;
      int bot = vr;
      while (bot + 1 < BOARD_SIZE && !oriented_at(bot + 1, vc, t).is_empty()) ++bot;
      if (top - 1 >= 0) cross[(top - 1) * BOARD_SIZE + vc] = cross_check_at(t, top - 1, vc);
      if (bot + 1 < BOARD_SIZE) cross[(bot + 1) * BOARD_SIZE + vc] = cross_check_at(t, bot + 1, vc);
    }
  }

  // A GADDAG anchor depends on a square and its four neighbors, so re-evaluate
  // each placed square and its neighbors.
  static const int dr[4] = {-1, 1, 0, 0};
  static const int dc[4] = {0, 0, -1, 1};
  for (int i = 0; i < n; ++i) {
    const int br = placed[i].first, bc = placed[i].second;
    for (int k = -1; k < 4; ++k) {
      const int ar = (k < 0) ? br : br + dr[k];
      const int ac = (k < 0) ? bc : bc + dc[k];
      if (!in_bounds(ar, ac)) continue;
      for (int t = 0; t < 2; ++t) {
        const int vr = t ? ac : ar;
        const int vc = t ? ar : ac;
        ganchor_[t][vr * BOARD_SIZE + vc] = gaddag_anchor_at(t, vr, vc);
      }
    }
  }
}

void Board::ensure_movegen_caches(const Dictionary& dict) const {
  if (caches_valid_ && dict_ == &dict) return;
  dict_ = &dict;
  recompute_all_caches();
  caches_valid_ = true;
}

}  // namespace scribblez
