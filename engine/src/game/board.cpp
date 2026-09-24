#include "game/board.h"

#include "game/bag.h"
#include "game/move.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
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

// Encoded as: ' ' NONE, '\'' DLS, '"' TLS, '-' DWS, '=' TWS (MAGPIE's
// bonus-square characters). The center star is a DWS.
// clang-format off
constexpr const char* kPremiumLayout[BOARD_SIZE] = {
    "=  '   =   '  =",
    " -   \"   \"   - ",
    "  -   ' '   -  ",
    "'  -   '   -  '",
    "    -     -    ",
    " \"   \"   \"   \" ",
    "  '   ' '   '  ",
    "=  '   -   '  =",
    "  '   ' '   '  ",
    " \"   \"   \"   \" ",
    "    -     -    ",
    "'  -   '   -  '",
    "  -   ' '   -  ",
    " -   \"   \"   - ",
    "=  '   =   '  =",
};
// clang-format on

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

void Board::apply(const Move& move) { apply(move, nullptr); }

void Board::apply(const Move& move, BoardUndo* undo) {
  DEBUG_ASSERT(move.transposed() == transposed_);
  if (undo) {
    undo->clear();
    undo->prev_caches_valid = caches_valid_;
  }
  if (move.type() != MoveType::PLAY) return;
  const bool had_caches = caches_valid_;
  const bool was_empty = empty_board();
  const bool horizontal = move.horizontal();
  const int start = move.start();
  std::array<std::pair<int, int>, RACK_SIZE> placed{};
  int np = 0;
  uint16_t mask = move.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (!in_bounds(r, c)) break;
    const int idx = r * BOARD_SIZE + c;
    if (undo) undo->squares.push_back({uint16_t(idx), squares_[idx]});
    set(r, c, move.glyph(np));  // clears caches_valid_
    placed[np++] = {r, c};
  }
  if (np == 0 || !had_caches) return;  // nothing placed, or caches were stale anyway

  // The first move replaces the lone center anchor with the tile-adjacency
  // anchors, so it gets a full rebuild; later moves update incrementally.
  recorder_ = undo;
  if (was_empty) {
    recompute_all_caches();
  } else {
    update_caches_after_place(placed.data(), np);
  }
  recorder_ = nullptr;
  caches_valid_ = true;
}

void Board::unapply(const BoardUndo& undo) {
  for (auto it = undo.crosses.rbegin(); it != undo.crosses.rend(); ++it)
    cross_[it->transposed][it->idx] = it->old;
  for (auto it = undo.anchors.rbegin(); it != undo.anchors.rend(); ++it)
    ganchor_[it->transposed][it->idx] = it->old;
  for (auto it = undo.squares.rbegin(); it != undo.squares.rend(); ++it) {
    Glyph& sq = squares_[it->idx];
    num_tiles_ += int(!it->old.is_empty()) - int(!sq.is_empty());
    sq = it->old;
  }
  caches_valid_ = undo.prev_caches_valid;
}

Board Board::transpose() const {
  Board out = *this;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) out.squares_[c * BOARD_SIZE + r] = at(r, c);
  // A cache entry in this board's transposed view is the same entry in the
  // transposed board's natural view, and vice versa.
  out.cross_[0] = cross_[1];
  out.cross_[1] = cross_[0];
  out.ganchor_[0] = ganchor_[1];
  out.ganchor_[1] = ganchor_[0];
  out.transposed_ = !transposed_;
  return out;
}

void Board::set_cross_(int transposed, int idx, const CrossCheck& cc) const {
  if (recorder_)
    recorder_->crosses.push_back({uint8_t(transposed), uint16_t(idx), cross_[transposed][idx]});
  cross_[transposed][idx] = cc;
}

void Board::set_anchor_(int transposed, int idx, bool value) const {
  if (recorder_)
    recorder_->anchors.push_back({uint8_t(transposed), uint16_t(idx), ganchor_[transposed][idx]});
  ganchor_[transposed][idx] = value;
}

std::string Board::to_string() const {
  std::string s;
  s.reserve((BOARD_SIZE + 1) * (BOARD_SIZE + 4));
  s += "   ";
  for (int c = 0; c < BOARD_SIZE; ++c) {
    s.push_back(char('A' + c));
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
        s.push_back(sq.is_blank() ? char('a' + sq.letter()) : char('A' + sq.letter()));
        s.push_back(' ');
      }
    }
    s.push_back('\n');
  }
  return s;
}

// ---- Move-generation caches ----

std::pair<int, int> Board::perpendicular_run_bounds(bool t, int r, int c) const {
  int top = r - 1;
  while (top >= 0 && !oriented_at(top, c, t).is_empty()) --top;
  ++top;
  int bot = r + 1;
  while (bot < BOARD_SIZE && !oriented_at(bot, c, t).is_empty()) ++bot;
  --bot;
  return {top, bot};
}

uint32_t Board::cross_check_letter_mask(bool t, int c, uint32_t prefix_node, int r, int bot) const {
  const Dictionary& dict = *dict_;
  uint32_t mask = 0;
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
  return mask;
}

CrossCheck Board::cross_check_at(bool t, int r, int c) const {
  CrossCheck cc;
  if (!oriented_at(r, c, t).is_empty()) return cc;  // filled squares: unused default

  auto [top, bot] = perpendicular_run_bounds(t, r, c);
  cc.has_neighbor = (top < r) || (bot > r);
  if (!cc.has_neighbor) {
    cc.mask = kAllLettersMask;
    cc.score = 0;
    return cc;
  }

  // Walk the DAWG through the run above (r, c), totalling the face value of the
  // run above and below.
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
  cc.mask = prefix_ok ? cross_check_letter_mask(t, c, prefix_node, r, bot) : 0;
  return cc;
}

bool Board::gaddag_anchor_at(bool t, int r, int c) const {
  // GADDAG anchors, one per place a play can start its leftward walk (this
  // view's rows only):
  //   - a filled square is an anchor iff it ends a run (nothing to its right);
  //   - an empty square is an anchor iff it touches no in-row tile but has one
  //     above or below (a pure hook). Squares next to an in-row run are
  //     reached from that run's anchor instead.
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
    for (int r = 0; r < BOARD_SIZE; ++r)
      for (int c = 0; c < BOARD_SIZE; ++c)
        set_cross_(t, r * BOARD_SIZE + c, cross_check_at(t, r, c));
    if (empty) {
      // Never reached under an undo recorder (apply() rebuilds only after
      // placing a tile), so the direct writes are safe.
      ganchor_[t].fill(false);
      ganchor_[t][CENTER * BOARD_SIZE + CENTER] = true;  // sole opening anchor
    } else {
      for (int r = 0; r < BOARD_SIZE; ++r)
        for (int c = 0; c < BOARD_SIZE; ++c)
          set_anchor_(t, r * BOARD_SIZE + c, gaddag_anchor_at(t, r, c));
    }
  }
}

void Board::update_caches_after_place(const std::pair<int, int>* placed, int n) const {
  for (int t = 0; t < 2; ++t) {
    for (int i = 0; i < n; ++i) {
      const int vr = t ? placed[i].second : placed[i].first;
      const int vc = t ? placed[i].first : placed[i].second;
      // The placed square is now filled; its cross-check is unused.
      set_cross_(t, vr * BOARD_SIZE + vc, CrossCheck{});
      // Only the empty squares just past each end of the perpendicular run
      // through the placed tile can change.
      int top = vr;
      while (top - 1 >= 0 && !oriented_at(top - 1, vc, t).is_empty()) --top;
      int bot = vr;
      while (bot + 1 < BOARD_SIZE && !oriented_at(bot + 1, vc, t).is_empty()) ++bot;
      if (top - 1 >= 0) set_cross_(t, (top - 1) * BOARD_SIZE + vc, cross_check_at(t, top - 1, vc));
      if (bot + 1 < BOARD_SIZE)
        set_cross_(t, (bot + 1) * BOARD_SIZE + vc, cross_check_at(t, bot + 1, vc));
    }
  }

  // An anchor depends on a square and its four neighbors, so re-evaluate each
  // placed square (k == -1) and its neighbors.
  for (int i = 0; i < n; ++i) {
    const int br = placed[i].first, bc = placed[i].second;
    for (int k = -1; k < 4; ++k) {
      const int ar = (k < 0) ? br : br + util::kFourNeighborDeltas[k].first;
      const int ac = (k < 0) ? bc : bc + util::kFourNeighborDeltas[k].second;
      if (!in_bounds(ar, ac)) continue;
      for (int t = 0; t < 2; ++t) {
        const int vr = t ? ac : ar;
        const int vc = t ? ar : ac;
        set_anchor_(t, vr * BOARD_SIZE + vc, gaddag_anchor_at(t, vr, vc));
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

TileCounts Board::tile_counts() const {
  TileCounts tiles;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = at(r, c);
      if (g.has_letter()) tiles.add(g.rack_tile());
    }
  }
  return tiles;
}

TileCounts Board::unseen_tiles(const Rack& held) const {
  TileCounts unseen = TileCounts::full_distribution();
  if (!unseen.remove(tile_counts()))
    throw util::Exception("board holds more copies of a tile than the distribution allows");
  if (!unseen.remove(held.counts()))
    throw util::Exception("the held rack holds a tile the distribution has run out of");
  return unseen;
}

int Board::unseen_count(int held_tiles) const {
  return Bag::kTotalTiles - num_tiles() - held_tiles;
}

int Board::pov_bag_size(int held_tiles) const {
  return std::max(0, unseen_count(held_tiles) - RACK_SIZE);
}

Rack Board::hidden_rack(const Rack& known) const {
  const TileCounts hidden_counts = unseen_tiles(known);
  if (hidden_counts.size() > RACK_SIZE)
    throw util::Exception("more than a rackful of tiles is unaccounted for: the bag is not empty");
  Rack hidden;
  for (int t = 0; t < TILE_KINDS; ++t) {
    for (int i = 0; i < hidden_counts.count(Tile::of(t)); ++i) hidden.add(Tile::of(t));
  }
  return hidden;
}

}  // namespace scribblez
