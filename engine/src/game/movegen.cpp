#include "game/movegen.h"

#include "util/math.h"

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// The board as seen by one scanning pass. The transposed view swaps rows and
// columns, so every generator only ever builds "horizontal" plays along a row.
struct View {
  const Board& board;
  bool transposed;
  Glyph at(int r, int c) const { return transposed ? board.at(c, r) : board.at(r, c); }
  Premium premium_at(int r, int c) const {
    return transposed ? board.premium_at(c, r) : board.premium_at(r, c);
  }
  std::pair<int, int> to_board(int r, int c) const {
    return transposed ? std::make_pair(c, r) : std::make_pair(r, c);
  }
};

// One view's Board-owned caches, indexed by idx(row, col).
using CrossChecks = std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>;
using Anchors = std::array<bool, BOARD_SIZE * BOARD_SIZE>;

constexpr int idx(int r, int c) { return r * BOARD_SIZE + c; }

// Whether a transposed-pass single-tile play duplicates one from the
// horizontal pass. A single tile that forms words along both axes is found by
// both passes; the horizontal pass owns it. A single tile whose only word lies
// along this pass has no twin and is kept. The word spans
// [start_col, end_col_excl) of `row`, and its one empty square is the placed
// tile.
bool single_tile_duplicates_horizontal_pass(const View& view, const CrossChecks& cross, int row,
                                            int start_col, int end_col_excl) {
  for (int c = start_col; c < end_col_excl; ++c) {
    if (view.at(row, c).is_empty()) return cross[idx(row, c)].has_neighbor;
  }
  return false;  // unreachable: a play places at least one tile
}

// Appel-Jacobson anchors: empty squares orthogonally adjacent to a tile, or
// the center on an empty board. The DAWG generator's counterpart to the
// Board's cached GADDAG anchors.
Anchors compute_anchors(const View& view) {
  Anchors anchor{};
  bool any_tile = false;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!view.at(r, c).is_empty()) {
        any_tile = true;
        break;
      }
    }
    if (any_tile) break;
  }
  if (!any_tile) {
    anchor[idx(CENTER, CENTER)] = true;
    return anchor;
  }
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!view.at(r, c).is_empty()) continue;
      for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
        int nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) continue;
        if (!view.at(nr, nc).is_empty()) {
          anchor[idx(r, c)] = true;
          break;
        }
      }
    }
  }
  return anchor;
}

// Builds and scores the play whose word occupies [start_col, end_col_excl) of
// `row`. Filled squares keep their letters; empty ones take `placed_letter` /
// `placed_blank`. Every generator scores through here, which is what makes
// their moves byte-identical.
Move build_play(const View& view, const CrossChecks& cross, int row, int start_col,
                int end_col_excl, const std::array<Tile, BOARD_SIZE>& placed_letter,
                const std::array<bool, BOARD_SIZE>& placed_blank) {
  std::array<Glyph, RACK_SIZE> played{};
  int n_placed = 0;
  uint16_t square_mask = 0;
  int main_letter_sum = 0;
  int word_mult = 1;
  int cross_total = 0;

  for (int c = start_col; c < end_col_excl; ++c) {
    Glyph sq = view.at(row, c);
    Tile L;
    bool is_blank;
    bool newly_placed;
    if (!sq.is_empty()) {
      L = sq.letter();
      is_blank = sq.is_blank();
      newly_placed = false;
    } else {
      L = placed_letter[c];
      is_blank = placed_blank[c];
      newly_placed = true;
      played[n_placed++] = Glyph::played(L, is_blank);
      square_mask |= uint16_t(1u << c);
    }

    int letter_value = is_blank ? 0 : TILE_VALUES[L];
    if (newly_placed) {
      Premium p = view.premium_at(row, c);
      letter_value *= p.letter_mult();
      word_mult *= p.word_mult();

      const CrossCheck& cc = cross[idx(row, c)];
      if (cc.has_neighbor) {
        int placed_v = is_blank ? 0 : TILE_VALUES[L];
        placed_v *= p.letter_mult();
        cross_total += (cc.score + placed_v) * p.word_mult();
      }
    }
    main_letter_sum += letter_value;
  }

  int score = main_letter_sum * word_mult + cross_total;
  if (n_placed == RACK_SIZE) score += 50;  // bingo
  return Move::play(!view.transposed, row, square_mask, uint16_t(score), played.data(), n_placed,
                    view.board.transposed());
}

// ---------------------------------------------------------------------------
// DAWG reference generator (Appel-Jacobson). From each anchor it chooses a
// left part, then extends right through the DAWG. The left part is either
// fresh tiles on the empty squares before the anchor (case A) or, when the
// anchor directly follows existing tiles, those tiles (case B). Kept only to
// cross-check the GADDAG generator in tests.
// ---------------------------------------------------------------------------
struct GenState {
  GenState(const View& view, const Dictionary& dict, const CrossChecks& cross,
           const Anchors& anchor, TileCounts rack, std::vector<Move>& out)
      : view(view), dict(dict), cross(cross), anchor(anchor), rack(std::move(rack)), out(out) {}

  const View& view;
  const Dictionary& dict;
  const CrossChecks& cross;
  const Anchors& anchor;
  TileCounts rack;  // tiles still available to the recursion
  std::vector<Move>& out;

  // The word under construction.
  std::vector<std::pair<Tile, bool>> left_letters;  // (letter, is_blank), left to right
  struct RightTile {
    int col;
    Tile letter;
    bool is_blank;
  };
  std::vector<RightTile> right_placed;
  int current_row = 0;
  int current_anchor_col = 0;
  int case_b_start_col = -1;  // case B: where the existing prefix begins; -1 in case A

  void emit_move(int start_col, int end_col_excl);

  void extend_right(int col, uint32_t node, bool accepts_here);
  void left_part(int limit, uint32_t node);
  void generate_for_row(int row);
};

void GenState::emit_move(int start_col, int end_col_excl) {
  std::array<Tile, BOARD_SIZE> placed_letter{};
  std::array<bool, BOARD_SIZE> placed_blank{};
  int li = 0, ri = 0;
  for (int c = start_col; c < end_col_excl; ++c) {
    if (!view.at(current_row, c).is_empty()) continue;
    if (case_b_start_col < 0 && c < current_anchor_col) {
      placed_letter[c] = left_letters[li].first;
      placed_blank[c] = left_letters[li].second;
      ++li;
    } else {
      placed_letter[c] = right_placed[ri].letter;
      placed_blank[c] = right_placed[ri].is_blank;
      ++ri;
    }
  }
  out.push_back(
    build_play(view, cross, current_row, start_col, end_col_excl, placed_letter, placed_blank));
}

void GenState::extend_right(int col, uint32_t node, bool accepts_here) {
  bool off_board = (col >= BOARD_SIZE);
  bool stop_here = off_board || view.at(current_row, col).is_empty();
  if (stop_here) {
    int total_placed = (int)left_letters.size() + (int)right_placed.size();
    if (accepts_here && col > current_anchor_col && total_placed > 0) {
      int start_col;
      if (case_b_start_col >= 0) {
        start_col = case_b_start_col;
      } else {
        start_col = current_anchor_col - (int)left_letters.size();
      }
      const bool suppress =
        view.transposed && total_placed == 1 &&
        single_tile_duplicates_horizontal_pass(view, cross, current_row, start_col, col);
      if (!suppress) emit_move(start_col, col);
    }
    if (off_board) return;
    const CrossCheck& cc = cross[idx(current_row, col)];
    uint32_t cmask = cc.mask;
    for (Tile L = Tile::of(0); L < 26; ++L) {
      if ((cmask & (1u << L)) == 0) continue;
      auto tr = dict.step(node, L);
      if (!tr.valid) continue;
      if (rack.count(L) > 0) {
        rack.remove(L);
        right_placed.push_back(RightTile{col, L, false});
        extend_right(col + 1, tr.next, tr.accepts);
        right_placed.pop_back();
        rack.add(L);
      }
      if (rack.blanks() > 0) {
        rack.remove(BLANK);
        right_placed.push_back(RightTile{col, L, true});
        extend_right(col + 1, tr.next, tr.accepts);
        right_placed.pop_back();
        rack.add(BLANK);
      }
    }
  } else {
    Tile L = view.at(current_row, col).letter();
    auto tr = dict.step(node, L);
    if (!tr.valid) return;
    extend_right(col + 1, tr.next, tr.accepts);
  }
}

void GenState::left_part(int limit, uint32_t node) {
  // Try ending the left part here. `accepts_here` doesn't matter: extend_right
  // never emits at the anchor column itself.
  extend_right(current_anchor_col, node, /*accepts_here=*/false);
  if (limit <= 0) return;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    auto tr = dict.step(node, L);
    if (!tr.valid) continue;
    if (rack.count(L) > 0) {
      rack.remove(L);
      left_letters.emplace_back(L, false);
      left_part(limit - 1, tr.next);
      left_letters.pop_back();
      rack.add(L);
    }
    if (rack.blanks() > 0) {
      rack.remove(BLANK);
      left_letters.emplace_back(L, true);
      left_part(limit - 1, tr.next);
      left_letters.pop_back();
      rack.add(BLANK);
    }
  }
}

void GenState::generate_for_row(int row) {
  current_row = row;
  for (int col = 0; col < BOARD_SIZE; ++col) {
    if (!anchor[idx(row, col)]) continue;
    current_anchor_col = col;

    if (col > 0 && !view.at(row, col - 1).is_empty()) {
      // Case B: the left part is the existing run ending at col - 1.
      int start_c = col - 1;
      while (start_c - 1 >= 0 && !view.at(row, start_c - 1).is_empty()) --start_c;
      uint32_t node = dict.root();
      bool ok = true;
      for (int x = start_c; x < col; ++x) {
        auto tr = dict.step(node, view.at(row, x).letter());
        if (!tr.valid) {
          ok = false;
          break;
        }
        node = tr.next;
      }
      if (ok) {
        case_b_start_col = start_c;
        extend_right(col, node, /*accepts_here=*/false);
        case_b_start_col = -1;
      }
    } else {
      // Case A: the left part may take the empty non-anchor squares to the left.
      int limit = 0;
      int c2 = col - 1;
      while (c2 >= 0 && view.at(row, c2).is_empty() && !anchor[idx(row, c2)]) {
        ++limit;
        --c2;
      }
      left_part(limit, dict.root());
    }
  }
}

// ---------------------------------------------------------------------------
// GADDAG generator (Gordon's algorithm). From each anchor it places tiles
// leftward along reversed-prefix arcs, then crosses the separator arc to place
// tiles rightward of the anchor. Leftward extension stops short of the
// previous anchor in the row, so each play is found from exactly one anchor.
// The recursion mirrors Gordon's Gen/GoOn.
// ---------------------------------------------------------------------------
struct GaddagGen {
  GaddagGen(const View& view, const Dictionary& dict, const CrossChecks& cross,
            const Anchors& anchor, TileCounts rack, std::vector<Move>& out)
      : view(view), dict(dict), cross(cross), anchor(anchor), rack(std::move(rack)), out(out) {}

  const View& view;
  const Dictionary& dict;
  const CrossChecks& cross;
  const Anchors& anchor;
  TileCounts rack;  // tiles still available to the recursion
  std::vector<Move>& out;

  int current_row = 0;
  int current_anchor_col = 0;
  int last_anchor_col = 100;  // 100: no previous anchor in this row
  int tiles_played = 0;
  std::array<Tile, BOARD_SIZE> strip_letter{};
  std::array<bool, BOARD_SIZE> strip_blank{};
  // The current row's cells, flattened once per row so the recursion indexes a
  // plain array instead of going through View::at()'s transpose branch.
  std::array<Glyph, BOARD_SIZE> row_cells{};

  void record(int leftstrip, int rightstrip) {
    if (view.transposed && tiles_played == 1 &&
        single_tile_duplicates_horizontal_pass(view, cross, current_row, leftstrip,
                                               rightstrip + 1)) {
      return;
    }
    out.push_back(
      build_play(view, cross, current_row, leftstrip, rightstrip + 1, strip_letter, strip_blank));
  }

  // Gordon's GoOn: letter L at `col` (placed or already there) led to
  // `new_node`; `accepts` says the path so far spells a word.
  void go_on(int col, Tile L, bool is_blank, uint32_t new_node, bool accepts, int leftstrip,
             int rightstrip) {
    const bool placed = row_cells[col].is_empty();
    if (placed) {
      strip_letter[col] = L;
      strip_blank[col] = is_blank;
    }

    if (col <= current_anchor_col) {
      leftstrip = col;
      const bool no_letter_left = (col == 0) || row_cells[col - 1].is_empty();
      if (accepts && no_letter_left && tiles_played > 0) {
        record(leftstrip, rightstrip);
      }
      if (new_node == 0) return;
      if (col > 0 && col - 1 != last_anchor_col) {
        recursive_gen(col - 1, new_node, leftstrip, rightstrip);
      }
      // Cross the separator to continue rightward from the anchor.
      auto sep = dict.step_tile(new_node, Dictionary::SEPARATOR);
      if (sep.valid && sep.next != 0 && no_letter_left && current_anchor_col < BOARD_SIZE - 1) {
        recursive_gen(current_anchor_col + 1, sep.next, leftstrip, rightstrip);
      }
    } else {
      rightstrip = col;
      const bool no_letter_right = (col == BOARD_SIZE - 1) || row_cells[col + 1].is_empty();
      if (accepts && no_letter_right && tiles_played > 0) {
        record(leftstrip, rightstrip);
      }
      if (new_node != 0 && col < BOARD_SIZE - 1) {
        recursive_gen(col + 1, new_node, leftstrip, rightstrip);
      }
    }
  }

  // Gordon's Gen: at board column `col`, GADDAG node `node`.
  void recursive_gen(int col, uint32_t node, int leftstrip, int rightstrip) {
    Glyph here = row_cells[col];
    if (!here.is_empty()) {
      // A tile is already here: follow its arc only.
      auto tr = dict.step(node, here.letter());
      if (tr.valid) {
        go_on(col, here.letter(), here.is_blank(), tr.next, tr.accepts, leftstrip, rightstrip);
      }
      return;
    }
    if (node == 0 || rack.empty()) return;
    const CrossCheck& cc = cross[idx(current_row, col)];
    // One pass over the arc list rather than a step() per letter.
    for (uint32_t i = node;; ++i) {
      uint32_t a = dict.arc(i);
      uint8_t tv = Dictionary::arc_tile(a);  // 0 = separator, 1..26 = A..Z
      if (tv >= 1 && tv <= 26 && (cc.mask & (1u << (tv - 1)))) {
        Tile L = Tile::of(tv - 1);
        uint32_t next = a & Dictionary::ARC_MASK;
        bool accepts = (a & Dictionary::ACCEPTS_BIT) != 0;
        if (rack.count(L) > 0) {
          rack.remove(L);
          ++tiles_played;
          go_on(col, L, false, next, accepts, leftstrip, rightstrip);
          --tiles_played;
          rack.add(L);
        }
        if (rack.blanks() > 0) {
          rack.remove(BLANK);
          ++tiles_played;
          go_on(col, L, true, next, accepts, leftstrip, rightstrip);
          --tiles_played;
          rack.add(BLANK);
        }
      }
      if (a & Dictionary::IS_END_BIT) break;
    }
  }

  void generate_for_row(int row) {
    current_row = row;
    last_anchor_col = 100;
    for (int col = 0; col < BOARD_SIZE; ++col) row_cells[col] = view.at(row, col);
    for (int col = 0; col < BOARD_SIZE; ++col) {
      if (!anchor[idx(row, col)]) continue;
      current_anchor_col = col;
      recursive_gen(col, dict.gaddag_root(), col, col);
      last_anchor_col = col;
    }
  }

  // The plays anchored at (row, col), identical to what generate_for_row finds
  // there. `last_anchor` is the previous anchor in the row (100 if none), which
  // generate_for_row would have set.
  void generate_one_anchor(int row, int col, int last_anchor) {
    current_row = row;
    for (int c = 0; c < BOARD_SIZE; ++c) row_cells[c] = view.at(row, c);
    last_anchor_col = last_anchor;
    current_anchor_col = col;
    tiles_played = 0;
    recursive_gen(col, dict.gaddag_root(), col, col);
  }
};

// ---------------------------------------------------------------------------
// Per-anchor shadow bounds (ShadowMoveGen::anchors).
// ---------------------------------------------------------------------------

// The rack's tile values sorted descending, blanks as 0.
std::vector<int> rack_values_desc(const TileCounts& rack) {
  std::vector<int> v;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    for (int i = 0; i < rack.count(L); ++i) v.push_back(TILE_VALUES[L]);
  }
  for (int i = 0; i < rack.blanks(); ++i) v.push_back(0);
  std::sort(v.begin(), v.end(), std::greater<int>());
  return v;
}

// One view row's inputs to anchor_score_bounds, built once per row.
struct LaneInfo {
  std::array<bool, BOARD_SIZE> filled{}, placeable{};
  std::array<int, BOARD_SIZE> tval{}, lmul{}, wmul{}, cscore{}, maxval{};
  std::array<int8_t, BOARD_SIZE> letter_idx{};  // A..Z index at filled squares
  std::array<bool, BOARD_SIZE> cneigh{};        // cross-word forms here
  std::array<int, BOARD_SIZE + 1> pref{};       // prefix sum of tval
  // At empty squares: lmul/wmul are premiums, cscore the cross-word's existing
  // score, placeable whether any rack tile fits, maxval the best such tile's
  // value. tval is the face value at filled squares.
};

// The letters (mask over A..Z) with an arc out of `node`, ignoring the
// separator: MAGPIE's KWG extension set.
uint32_t letter_children_mask(const Dictionary& dict, uint32_t node) {
  uint32_t mask = 0;
  for (uint32_t i = node;; ++i) {
    const uint32_t arc = dict.arc(i);
    const uint8_t tv = Dictionary::arc_tile(arc);
    if (tv >= 1 && tv <= 26) mask |= (1u << (tv - 1));
    if (arc & Dictionary::IS_END_BIT) break;
  }
  return mask;
}

// In-row extension sets, as MAGPIE's game_gen_classic_cross_set computes them.
// For each run of tiles [a, b], walking the GADDAG through the run reversed
// gives a node whose letter arcs are the letters that can precede the run, and
// whose separator arc leads to the letters that can follow it.
//   - left_ext: at a - 1 (the square before the run) and at b (the run's end,
//     where its anchor sits), the letters that can precede the run.
//   - right_ext: at b, the letters that can follow the run.
// Squares next to no run get all letters. Any real play must at least extend
// the adjacent run, so pruning with these sets never drops a play.
void compute_lane_extensions(const View& view, const Dictionary& dict, int row,
                             std::array<uint32_t, BOARD_SIZE>& left_ext,
                             std::array<uint32_t, BOARD_SIZE>& right_ext) {
  left_ext.fill(kAllLettersMask);
  right_ext.fill(kAllLettersMask);
  for (int a = 0; a < BOARD_SIZE;) {
    if (view.at(row, a).is_empty()) {
      ++a;
      continue;
    }
    int b = a;
    while (b + 1 < BOARD_SIZE && !view.at(row, b + 1).is_empty()) ++b;
    uint32_t node = dict.gaddag_root();
    bool ok = true;
    for (int k = b; k >= a && ok; --k) {
      const Dictionary::Step s = dict.step(node, view.at(row, k).letter());
      ok = s.valid;
      node = s.next;
    }
    uint32_t es_left = 0, es_right = 0;
    if (ok && node != 0) {
      es_left = letter_children_mask(dict, node);
      const Dictionary::Step sep = dict.step_tile(node, Dictionary::SEPARATOR);
      if (sep.valid && sep.next != 0) es_right = letter_children_mask(dict, sep.next);
    }
    left_ext[b] = es_left;
    if (a > 0) left_ext[a - 1] = es_left;
    right_ext[b] = es_right;
    a = b + 1;
  }
}

void build_lane(const View& view, const CrossChecks& cross, int row, uint32_t rack_letter_mask,
                bool has_blank, const std::array<uint32_t, BOARD_SIZE>& inlane_ext,
                LaneInfo& lane) {
  lane = LaneInfo{};
  for (int c = 0; c < BOARD_SIZE; ++c) {
    const Glyph g = view.at(row, c);
    if (!g.is_empty()) {
      lane.filled[c] = true;
      lane.tval[c] = g.is_blank() ? 0 : TILE_VALUES[g.letter()];
      lane.letter_idx[c] = int8_t(g.letter().index());
      continue;
    }
    const Premium p = view.premium_at(row, c);
    lane.lmul[c] = p.letter_mult();
    lane.wmul[c] = p.word_mult();
    const CrossCheck& cc = cross[idx(row, c)];
    lane.cscore[c] = cc.score;
    lane.cneigh[c] = cc.has_neighbor;
    // A square no rack tile can fill makes any window covering it infeasible.
    const uint32_t playable = cc.mask & inlane_ext[c];
    const uint32_t allowed = playable & rack_letter_mask;
    if (allowed != 0) {
      lane.placeable[c] = true;
      int mv = 0;
      for (Tile L = Tile::of(0); L < 26; ++L) {
        if ((allowed & (1u << L)) && TILE_VALUES[L] > mv) mv = TILE_VALUES[L];
      }
      lane.maxval[c] = mv;
    } else if (has_blank && playable != 0) {
      lane.placeable[c] = true;
      lane.maxval[c] = 0;
    }
  }
  for (int c = 0; c < BOARD_SIZE; ++c) lane.pref[c + 1] = lane.pref[c] + lane.tval[c];
}

// Fills out[e] with an upper bound on the score of any play from anchor `col`
// placing e tiles (-1 if there is none). Enumerates every window [a, b] that
// covers the anchor without reaching the previous anchor. For each, the placed
// tiles' contribution is bounded by the smaller of two overestimates:
//   - the top e rack values paired with the window's best letter multipliers
//     (tight on tile counts);
//   - each square's best permitted tile times its multiplier (tight on
//     cross-checks).
// That, plus the existing tiles, times the word multipliers, plus cross-word
// and bingo terms that also overestimate. A true upper bound is what makes
// best-first pruning exact.
void anchor_score_bounds(const LaneInfo& lane, int col, int last_anchor_col,
                         const std::vector<int>& rack_vals_desc,
                         std::array<int, kMaxPlayTiles + 1>& out) {
  out.fill(-1);
  const int rack_size = rack_vals_desc.size();
  if (rack_size == 0) return;
  const int a_min = std::max(0, last_anchor_col + 1);
  const int e_cap = std::min(rack_size, kMaxPlayTiles);
  for (int a = col; a >= a_min; --a) {
    int e = 0, c2 = 0, c3 = 0, bad = 0, cross_sum = 0, psm = 0;
    long wprod = 1;
    auto add_square = [&](int c) {
      if (lane.filled[c]) return;
      ++e;
      if (!lane.placeable[c]) {
        ++bad;
        return;
      }
      const int lm = lane.lmul[c];
      if (lm >= 3)
        ++c3;
      else if (lm == 2)
        ++c2;
      wprod *= lane.wmul[c];
      psm += lane.maxval[c] * lm;
      if (lane.cneigh[c]) cross_sum += (lane.cscore[c] + lane.maxval[c] * lm) * lane.wmul[c];
    };
    for (int c = a; c < col; ++c) add_square(c);
    for (int b = col; b < BOARD_SIZE; ++b) {
      add_square(b);
      if (e == 0) continue;
      if (e > e_cap) break;  // e and bad only grow with b
      if (bad > 0) break;
      int wl = a;
      while (wl - 1 >= 0 && lane.filled[wl - 1]) --wl;
      int wr = b;
      while (wr + 1 < BOARD_SIZE && lane.filled[wr + 1]) ++wr;
      const int existing = lane.pref[wr + 1] - lane.pref[wl];
      int greedy = 0, taken = 0;
      for (int j = 0; j < c3 && taken < e; ++j) greedy += rack_vals_desc[taken++] * 3;
      for (int j = 0; j < c2 && taken < e; ++j) greedy += rack_vals_desc[taken++] * 2;
      while (taken < e) greedy += rack_vals_desc[taken++];
      const int placed = std::min(greedy, psm);
      const long main_word = long(placed + existing) * wprod;
      const long sc = main_word + cross_sum + (e == RACK_SIZE ? 50 : 0);
      if (sc > out[e]) out[e] = int(sc);
    }
  }
}

// ---------------------------------------------------------------------------
// WordMap generation.
// ---------------------------------------------------------------------------

// Buckets every non-empty sub-multiset of `letters` ((letter, count) pairs)
// into out[size].
void enum_subracks(const std::vector<std::pair<int, int>>& letters, size_t i, BitRack cur, int size,
                   std::array<std::vector<BitRack>, kMaxPlayTiles + 1>& out) {
  if (i == letters.size()) {
    if (size >= 1) out[size].push_back(cur);
    return;
  }
  const int li = letters[i].first;
  const int cnt = letters[i].second;
  for (int use = 0; use <= cnt; ++use) {
    enum_subracks(letters, i + 1, cur, size + use, out);
    cur.add_letter(li);
  }
}

// One view row's tiles, snapshotted for WordMap generation.
struct WmpLane {
  const View& view;
  const CrossChecks& cross;
  int row;
  std::array<bool, BOARD_SIZE> filled{};
  std::array<Tile, BOARD_SIZE> letter{};
};

WmpLane build_wmp_lane(const View& view, const CrossChecks& cross, int row) {
  WmpLane lane{view, cross, row, {}, {}};
  for (int c = 0; c < BOARD_SIZE; ++c) {
    const Glyph g = view.at(row, c);
    lane.filled[c] = !g.is_empty();
    if (lane.filled[c]) lane.letter[c] = g.letter();
  }
  return lane;
}

// The rack's letters as a mask over A..Z, read off the size-1 subracks.
uint32_t wmp_rack_letter_mask(const WmpSubracks& subracks) {
  uint32_t mask = 0;
  for (const BitRack& s : subracks[1]) {
    for (int l = 0; l < 26; ++l) {
      if (s.get(l)) mask |= 1u << l;
    }
  }
  return mask;
}

// Squares a word can cover: filled ones, and empty ones whose cross-check
// admits a rack letter. Skipping spans over other squares recovers the pruning
// the GADDAG gets for free by following arcs.
std::array<bool, BOARD_SIZE> wmp_placeable_squares(const WmpLane& lane, uint32_t rack_mask) {
  std::array<bool, BOARD_SIZE> placeable{};
  for (int c = 0; c < BOARD_SIZE; ++c) {
    placeable[c] = lane.filled[c] || (lane.cross[idx(lane.row, c)].mask & rack_mask) != 0;
  }
  return placeable;
}

// Appends the play of `word` (length L) at column `wl` if it fits: letters on
// filled squares must match and placed letters must pass cross-checks. Every
// WordMap play passes through here, so the single-tile dedup rule does too.
void wmp_try_word(const WmpLane& lane, int wl, int L, const Tile* word, std::vector<Move>& out) {
  std::array<Tile, BOARD_SIZE> placed_letter{};
  int placed = 0;
  for (int i = 0; i < L; ++i) {
    const int c = wl + i;
    if (lane.filled[c]) {
      if (word[i].index() != lane.letter[c].index()) return;
    } else if (lane.cross[idx(lane.row, c)].mask & (1u << word[i].index())) {
      placed_letter[c] = word[i];
      ++placed;
    } else {
      return;
    }
  }
  if (lane.view.transposed && placed == 1 &&
      single_tile_duplicates_horizontal_pass(lane.view, lane.cross, lane.row, wl, wl + L)) {
    return;
  }
  const std::array<bool, BOARD_SIZE> no_blanks{};
  out.push_back(build_play(lane.view, lane.cross, lane.row, wl, wl + L, placed_letter, no_blanks));
}

// Emits the plays of length L at column `wl` that place one of the given
// subracks.
void wmp_emit_span(const WmpLane& lane, const WordMap& wm, int wl, int L,
                   const BitRack& playthrough, const std::vector<BitRack>& subracks_of_size,
                   std::vector<Move>& out) {
  for (const BitRack& sub : subracks_of_size) {
    const WordMap::WordList words = wm.lookup(L, playthrough + sub);
    for (int wi = 0; wi < words.count; ++wi) {
      wmp_try_word(lane, wl, L, words.begin + wi * L, out);
    }
  }
}

// Every play whose leftmost placed tile is at `A`. Keying plays by that square
// partitions them, so full-board generation finds each play exactly once.
void wmp_emit_leftmost_anchor(const WmpLane& lane, const WordMap& wm, const WmpSubracks& subracks,
                              int rack_tiles, int A, bool empty_board, std::vector<Move>& out) {
  if (lane.filled[A]) return;
  int wl = A;
  while (wl - 1 >= 0 && lane.filled[wl - 1]) --wl;
  BitRack playthrough{};
  for (int c = wl; c < A; ++c) playthrough.add_letter(lane.letter[c].index());
  int placed = 0;
  bool any_playthrough = (wl < A);
  bool any_cross = false;
  for (int wr = A; wr < BOARD_SIZE; ++wr) {
    if (lane.filled[wr]) {
      playthrough.add_letter(lane.letter[wr].index());
      any_playthrough = true;
    } else {
      ++placed;
      if (lane.cross[idx(lane.row, wr)].has_neighbor) any_cross = true;
    }
    if (placed > rack_tiles) break;
    if (wr + 1 < BOARD_SIZE && lane.filled[wr + 1]) continue;  // word can't end here
    const int L = wr - wl + 1;
    if (L < 2) continue;
    // First move covers the center square; later moves must touch the board.
    const bool connected = empty_board ? (lane.row == CENTER && A <= CENTER && wr >= CENTER)
                                       : (any_playthrough || any_cross);
    if (!connected) continue;
    wmp_emit_span(lane, wm, wl, L, playthrough, subracks[placed], out);
  }
}

}  // namespace

MoveGenerator::MoveGenerator(const Board& board, const Dictionary& dict)
    : board_(board), dict_(dict) {}

std::vector<Move> MoveGenerator::generate(const Rack& rack, GenAlgo algo) {
  board_.ensure_movegen_caches(dict_);
  std::vector<Move> out;
  for (int orient = 0; orient < 2; ++orient) {
    bool transposed = (orient == 1);
    View view{board_, transposed};
    const CrossChecks& cross = board_.cross_checks(transposed);
    if (algo == GenAlgo::GADDAG) {
      const Anchors& anchors = board_.gaddag_anchors(transposed);
      GaddagGen st{view, dict_, cross, anchors, rack.counts(), out};
      for (int r = 0; r < BOARD_SIZE; ++r) st.generate_for_row(r);
    } else {
      auto anchors = compute_anchors(view);
      GenState st{view, dict_, cross, anchors, rack.counts(), out};
      for (int r = 0; r < BOARD_SIZE; ++r) st.generate_for_row(r);
    }
  }
  return out;
}

void MoveGenerator::generate_lane(const Rack& rack, bool transposed, int row,
                                  std::vector<Move>& out) {
  board_.ensure_movegen_caches(dict_);
  const View view{board_, transposed};
  GaddagGen st{
    view,          dict_, board_.cross_checks(transposed), board_.gaddag_anchors(transposed),
    rack.counts(), out};
  st.generate_for_row(row);
}

void wmp_rack_subracks(const Rack& rack, WmpSubracks& out, int& rack_tiles) {
  const TileCounts& counts = rack.counts();
  std::vector<std::pair<int, int>> letters;
  rack_tiles = 0;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    const int n = counts.count(L);
    if (n > 0) {
      letters.emplace_back(L.index(), n);
      rack_tiles += n;
    }
  }
  for (auto& bucket : out) bucket.clear();
  enum_subracks(letters, 0, BitRack{}, 0, out);
}

std::vector<Move> wmp_generate(const Board& board, const Dictionary& dict, const WordMap& wm,
                               const Rack& rack) {
  board.ensure_movegen_caches(dict);
  WmpSubracks subracks;
  int rack_tiles = 0;
  wmp_rack_subracks(rack, subracks, rack_tiles);
  const bool empty_board = board.empty_board();

  std::vector<Move> out;
  for (int orient = 0; orient < 2; ++orient) {
    const View view{board, orient == 1};
    const CrossChecks& cross = board.cross_checks(view.transposed);
    for (int r = 0; r < BOARD_SIZE; ++r) {
      const WmpLane lane = build_wmp_lane(view, cross, r);
      for (int A = 0; A < BOARD_SIZE; ++A) {
        wmp_emit_leftmost_anchor(lane, wm, subracks, rack_tiles, A, empty_board, out);
      }
    }
  }
  return out;
}

void wmp_generate_anchor(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         int rack_tiles, const ShadowAnchor& a, std::vector<Move>& out) {
  const View view{board, a.transposed};
  const CrossChecks& cross = board.cross_checks(a.transposed);
  const WmpLane lane = build_wmp_lane(view, cross, a.row);
  const std::array<bool, BOARD_SIZE> placeable =
    wmp_placeable_squares(lane, wmp_rack_letter_mask(subracks));
  const int col = a.col;
  const int left_limit = (a.last_anchor_col < 0) ? 0 : a.last_anchor_col + 1;

  // The word covers the anchor and starts at some wl in [left_limit, col], the
  // same range the GADDAG's leftward walk covers. The anchor square may be
  // filled (the end of a run) or empty.
  for (int wl = left_limit; wl <= col; ++wl) {
    if (wl > 0 && lane.filled[wl - 1]) continue;  // not a maximal word start
    BitRack playthrough{};
    int placed = 0;
    bool left_ok = true;
    for (int c = wl; c < col; ++c) {
      if (lane.filled[c]) {
        playthrough.add_letter(lane.letter[c].index());
      } else {
        ++placed;
        if (!placeable[c]) left_ok = false;
      }
    }
    if (!left_ok) continue;
    for (int wr = col; wr < BOARD_SIZE; ++wr) {
      if (lane.filled[wr]) {
        playthrough.add_letter(lane.letter[wr].index());
      } else {
        ++placed;
        if (!placeable[wr]) break;  // so is every longer span
      }
      if (placed > rack_tiles) break;
      if (wr + 1 < BOARD_SIZE && lane.filled[wr + 1]) continue;  // word can't end here
      const int L = wr - wl + 1;
      if (L < 2 || placed < 1) continue;
      wmp_emit_span(lane, wm, wl, L, playthrough, subracks[placed], out);
    }
  }
}

void wmp_generate_extent(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         const ShadowExtent& e, std::vector<Move>& out, double best_equity,
                         const double* sub_terms) {
  const View view{board, e.transposed};
  const CrossChecks& cross = board.cross_checks(e.transposed);
  const WmpLane lane = build_wmp_lane(view, cross, e.row);
  // As MAGPIE's wordmap_gen: one lookup per subrack, with each word found tried
  // at every start column. The sub_terms skip mirrors MAGPIE's
  // better_play_has_been_found check, and is sound because score_bound
  // overestimates every play the subrack could make.
  const std::vector<BitRack>& subs = subracks[e.placed];
  for (size_t j = 0; j < subs.size(); ++j) {
    if (sub_terms != nullptr && double(e.score_bound) + sub_terms[j] < best_equity) continue;
    const WordMap::WordList words = wm.lookup(e.length, e.pt + subs[j]);
    for (int wi = 0; wi < words.count; ++wi) {
      const Tile* word = words.begin + wi * e.length;
      for (int s = e.leftmost_start_col; s <= e.rightmost_start_col; ++s) {
        wmp_try_word(lane, s, e.length, word, out);
      }
    }
  }
}

ShadowMoveGen::ShadowMoveGen(const Board& board, const Dictionary& dict)
    : board_(board), dict_(dict) {}

std::vector<ShadowAnchor> ShadowMoveGen::anchors(const Rack& rack) const {
  board_.ensure_movegen_caches(dict_);
  const TileCounts& counts = rack.counts();
  const std::vector<int> rack_vals = rack_values_desc(counts);
  uint32_t rack_letter_mask = 0;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    if (counts.count(L) > 0) rack_letter_mask |= (1u << L);
  }
  const bool has_blank = counts.blanks() > 0;

  // No extension-set pruning: the bounds stay valid, just looser, and the
  // GADDAG enforces in-row validity during generation.
  std::array<uint32_t, BOARD_SIZE> trivial_ext;
  trivial_ext.fill(kAllLettersMask);

  std::vector<ShadowAnchor> out;
  LaneInfo lane;
  for (int orient = 0; orient < 2; ++orient) {
    const bool transposed = (orient == 1);
    View view{board_, transposed};
    const CrossChecks& cross = board_.cross_checks(transposed);
    const Anchors& anchors = board_.gaddag_anchors(transposed);
    for (int r = 0; r < BOARD_SIZE; ++r) {
      bool any = false;
      for (int c = 0; c < BOARD_SIZE && !any; ++c) any = anchors[idx(r, c)];
      if (!any) continue;
      build_lane(view, cross, r, rack_letter_mask, has_blank, trivial_ext, lane);
      int prev = -1;
      for (int c = 0; c < BOARD_SIZE; ++c) {
        if (!anchors[idx(r, c)]) continue;
        ShadowAnchor sa{transposed, r, c, prev, {}};
        anchor_score_bounds(lane, c, prev, rack_vals, sa.score_bound_by_size);
        out.push_back(sa);
        prev = c;
      }
    }
  }
  return out;
}

void ShadowMoveGen::generate_anchor(const ShadowAnchor& a, const Rack& rack,
                                    std::vector<Move>& out) const {
  View view{board_, a.transposed};
  const CrossChecks& cross = board_.cross_checks(a.transposed);
  const Anchors& anchors = board_.gaddag_anchors(a.transposed);
  GaddagGen st{view, dict_, cross, anchors, rack.counts(), out};
  st.generate_one_anchor(a.row, a.col, a.last_anchor_col < 0 ? 100 : a.last_anchor_col);
}

namespace {

// ---------------------------------------------------------------------------
// Per-extent shadow bounds (ShadowMoveGen::extents): a translation of MAGPIE's
// shadow walk in move_gen.c. Names follow MAGPIE's so the two can be read side
// by side; MAGPIE's source is the reference for the finer points.
// ---------------------------------------------------------------------------

// kRackAlign sizes the multiplier arrays as MAGPIE does. The anchor table has
// one slot per (playthrough_blocks, tiles_played).
constexpr int kRackAlign = util::align_up(RACK_SIZE, 8);
constexpr int kMaxPlaythroughBlocks = (BOARD_SIZE / 2) + 1;
constexpr int kMaxShadowAnchors = (RACK_SIZE + 1) * kMaxPlaythroughBlocks;
constexpr uint32_t kTrivialCrossSet = kAllLettersMask;

// One view row prepared for the shadow walk (MAGPIE's row_cache).
struct ShadowLane {
  std::array<bool, BOARD_SIZE> empty{};
  std::array<int8_t, BOARD_SIZE> letter{};       // A..Z index at filled squares
  std::array<uint32_t, BOARD_SIZE> cross_set{};  // 0 at filled squares
  std::array<int, BOARD_SIZE> cross_score{};
  std::array<bool, BOARD_SIZE> is_cross_word{};
  std::array<int, BOARD_SIZE> letter_mult{}, word_mult{};
  std::array<uint32_t, BOARD_SIZE> left_ext{}, right_ext{};
  std::array<bool, BOARD_SIZE> anchor{};
};

ShadowLane build_shadow_lane(const View& view, const CrossChecks& cross, const Anchors& anchors,
                             const Dictionary& dict, int row) {
  ShadowLane lane;
  compute_lane_extensions(view, dict, row, lane.left_ext, lane.right_ext);
  for (int c = 0; c < BOARD_SIZE; ++c) {
    const Glyph g = view.at(row, c);
    lane.empty[c] = g.is_empty();
    lane.anchor[c] = anchors[idx(row, c)];
    if (!lane.empty[c]) {
      lane.letter[c] = int8_t(g.letter().index());
      lane.cross_set[c] = 0;  // a filled square admits no fresh tile
      continue;
    }
    const CrossCheck& cc = cross[idx(row, c)];
    lane.letter[c] = -1;
    lane.cross_set[c] = cc.mask;
    lane.cross_score[c] = cc.score;
    lane.is_cross_word[c] = cc.has_neighbor;
    const Premium p = view.premium_at(row, c);
    lane.letter_mult[c] = p.letter_mult();
    lane.word_mult[c] = p.word_mult();
  }
  return lane;
}

// The letters of the first `blocks` runs of tiles at or after
// `rightmost_start_col`: an extent's playthrough multiset (MAGPIE's
// wmp_move_gen_set_playthrough_bit_rack).
BitRack set_playthrough_bitrack(const ShadowLane& lane, int rightmost_start_col, int blocks) {
  BitRack pt;
  if (blocks == 0) return pt;
  bool in_block = false;
  int found = 0;
  for (int col = rightmost_start_col; col < BOARD_SIZE; ++col) {
    if (lane.empty[col]) {
      if (in_block) {
        if (found == blocks) break;
        in_block = false;
      }
      continue;
    }
    pt.add_letter(lane.letter[col]);
    if (!in_block) {
      in_block = true;
      ++found;
    }
  }
  return pt;
}

// The shadow walk for one anchor. It grows a window left and right from the
// anchor, as the GADDAG would, and at each step bounds the score of any play
// covering that window. The maxima, keyed by (playthrough_blocks,
// tiles_played), land in the anchor table `slots`; each becomes a
// ShadowExtent.
//
// The bound: a square whose constraints admit exactly one rack letter is
// "restricted" and scores that letter exactly. The remaining tiles are
// assumed to land in the best way possible: the highest tile values paired
// with the highest effective multipliers of the unrestricted squares. A
// square's effective multiplier is its letter multiplier times the main word's
// multiplier, plus its multiplier within any cross-word it forms.
struct ShadowGen {
  const ShadowLane& lane;
  bool transposed;

  // Rack state, blanks excluded. Restricting a tile removes it from `rack`,
  // `rack_cross_set` and `descending_tile_scores`; the *_copy members save
  // state for shadow_play_right to restore.
  TileCounts rack;
  TileCounts full_rack;
  TileCounts player_rack_shadow_right_copy;
  uint32_t rack_cross_set = 0;
  int number_of_letters_on_rack = 0;
  std::array<int, RACK_SIZE> full_rack_descending{};
  std::array<int, RACK_SIZE> descending_tile_scores{};
  std::array<int, RACK_SIZE> descending_tile_scores_copy{};

  // Effective multipliers of the unrestricted squares, sorted descending.
  std::array<int, kRackAlign> descending_effective_letter_multipliers{};
  std::array<int, kRackAlign> desc_eff_letter_muls_copy{};
  struct XwMul {
    int multiplier;
    int column;
  };
  std::array<XwMul, kRackAlign> descending_cross_word_multipliers{};
  std::array<XwMul, kRackAlign> desc_xw_muls_copy{};
  int num_unrestricted_multipliers = 0;
  int last_word_multiplier = 1;

  // Restricted tiles' main-word score (before the word multiplier), and all
  // cross-word score.
  int shadow_mainword_restricted_score = 0;
  int shadow_perpendicular_additional_score = 0;
  int shadow_word_multiplier = 1;

  int max_tiles_to_play = 0;
  int tiles_played = 0;
  int current_left_col = 0, current_right_col = 0;
  int current_anchor_col = 0, last_anchor_col = 0;
  uint32_t anchor_left_extension_set = 0, anchor_right_extension_set = 0;

  // Existing tiles inside the window. Block count and tile count give the slot
  // key and word length; the multiset feeds the word-existence prune.
  int playthrough_blocks = 0, num_tiles_played_through = 0;
  BitRack playthrough_bit_rack;
  int playthrough_blocks_copy = 0, num_tiles_played_through_copy = 0;
  BitRack playthrough_bit_rack_copy;

  // Word-existence pruning; a null `wm` disables it.
  const WordMap* wm = nullptr;
  BitRack full_rack_bit_rack;
  std::array<bool, RACK_SIZE + 1> nonplaythrough_has_word{};

  // The anchor table; `touched` lists the slots written this anchor.
  struct Slot {
    int tiles_to_play;
    int playthrough_blocks;
    int word_length;
    int leftmost;
    int rightmost;
    int score;
  };
  std::array<Slot, kMaxShadowAnchors> slots{};
  std::array<int, kMaxShadowAnchors> touched{};
  int num_touched = 0;

  ShadowGen(const ShadowLane& lane, bool transposed, const TileCounts& counts, const WordMap* wm,
            const std::array<bool, RACK_SIZE + 1>& has_word)
      : lane(lane),
        transposed(transposed),
        rack(counts),
        full_rack(counts),
        wm(wm),
        nonplaythrough_has_word(has_word) {
    number_of_letters_on_rack = counts.size();
    int n = 0;
    for (Tile L = Tile::of(0); L < 26; ++L) {
      const int cnt = counts.count(L);
      if (cnt > 0) rack_cross_set |= (1u << L);
      for (int i = 0; i < cnt; ++i) {
        if (n < RACK_SIZE) full_rack_descending[n++] = TILE_VALUES[L];
        full_rack_bit_rack.add_letter(L.index());
      }
    }
    std::sort(full_rack_descending.begin(), full_rack_descending.begin() + n, std::greater<int>());
  }

  // tiles_to_play == 0 marks a slot unused this anchor; maybe_update_anchor
  // initializes the rest on first touch. So resetting only that field, and
  // only in touched slots, suffices.
  void reset_anchors() {
    for (int i = 0; i < num_touched; ++i) slots[touched[i]].tiles_to_play = 0;
    num_touched = 0;
  }

  void maybe_update_anchor(int tp, int word_length, int start_col, int score) {
    const int s = playthrough_blocks * (RACK_SIZE + 1) + tp;
    Slot& a = slots[s];
    a.playthrough_blocks = playthrough_blocks;
    a.word_length = word_length;
    if (a.tiles_to_play == 0) {
      touched[num_touched++] = s;
      a.tiles_to_play = tp;
      a.leftmost = a.rightmost = start_col;
      a.score = score;
      return;
    }
    a.tiles_to_play = tp;
    if (start_col < a.leftmost) a.leftmost = start_col;
    if (start_col > a.rightmost) a.rightmost = start_col;
    if (score > a.score) a.score = score;
  }

  void remove_score_from_descending_tile_scores(int score) {
    const int num_available = rack.size();
    for (int i = num_available; i-- > 0;) {
      if (descending_tile_scores[i] == score) {
        for (int j = i; j < num_available; ++j)
          descending_tile_scores[j] = descending_tile_scores[j + 1];
        descending_tile_scores[num_available] = 0;
        break;
      }
    }
  }

  void restrict_tile_and_accumulate_score(uint32_t possible, int letter_mult, int this_word_mult,
                                          int col) {
    const int ml = std::countr_zero(possible);
    rack.remove(Tile::of(ml));
    if (rack.count(Tile::of(ml)) == 0) rack_cross_set &= ~possible;
    const int tile_score = TILE_VALUES[ml];
    remove_score_from_descending_tile_scores(tile_score);
    const int lsm = tile_score * letter_mult;
    shadow_mainword_restricted_score += lsm;
    if (lane.is_cross_word[col]) shadow_perpendicular_additional_score += lsm * this_word_mult;
  }

  bool try_restrict_tile(uint32_t possible, int letter_mult, int this_word_mult, int col) {
    if (!std::has_single_bit(possible)) return false;
    restrict_tile_and_accumulate_score(possible, letter_mult, this_word_mult, col);
    return true;
  }

  void insert_unrestricted_cross_word_multiplier(int multiplier, int col) {
    int i = num_unrestricted_multipliers;
    for (; i > 0 && descending_cross_word_multipliers[i - 1].multiplier < multiplier; --i)
      descending_cross_word_multipliers[i] = descending_cross_word_multipliers[i - 1];
    descending_cross_word_multipliers[i] = XwMul{multiplier, col};
  }

  void insert_unrestricted_effective_letter_multiplier(int multiplier) {
    int i = num_unrestricted_multipliers;
    for (; i > 0 && descending_effective_letter_multipliers[i - 1] < multiplier; --i)
      descending_effective_letter_multipliers[i] = descending_effective_letter_multipliers[i - 1];
    descending_effective_letter_multipliers[i] = multiplier;
  }

  void maybe_recalculate_effective_multipliers() {
    if (last_word_multiplier == shadow_word_multiplier) return;
    last_word_multiplier = shadow_word_multiplier;
    const int orig = num_unrestricted_multipliers;
    num_unrestricted_multipliers = 0;
    for (int i = 0; i < orig; ++i) {
      const int xw = descending_cross_word_multipliers[i].multiplier;
      const int col = descending_cross_word_multipliers[i].column;
      const int eff = shadow_word_multiplier * lane.letter_mult[col] + xw;
      insert_unrestricted_effective_letter_multiplier(eff);
      ++num_unrestricted_multipliers;
    }
  }

  void insert_unrestricted_multipliers(int col) {
    maybe_recalculate_effective_multipliers();
    const int is_cross_word = lane.is_cross_word[col] ? 1 : 0;
    const int this_word_mult = lane.word_mult[col];
    const int letter_mult = lane.letter_mult[col];
    const int eff_xw = letter_mult * this_word_mult * is_cross_word;
    insert_unrestricted_cross_word_multiplier(eff_xw, col);
    const int main_word_mult = shadow_word_multiplier * letter_mult;
    insert_unrestricted_effective_letter_multiplier(main_word_mult + eff_xw);
    ++num_unrestricted_multipliers;
  }

  void shadow_record() {
    const int word_length = num_tiles_played_through + tiles_played;

    // Skip slots the WordMap proves hold no word, as MAGPIE's shadow_record
    // does. A play of k tiles with no playthrough needs some size-k subrack to
    // be a word; a full-rack play through tiles needs rack + playthrough to be
    // one. Partial-rack playthrough plays aren't checked.
    if (wm != nullptr) {
      if (num_tiles_played_through > 0) {
        if (tiles_played == number_of_letters_on_rack &&
            wm->lookup(word_length, playthrough_bit_rack + full_rack_bit_rack).count == 0)
          return;
      } else if (tiles_played >= 2 && !nonplaythrough_has_word[tiles_played]) {
        return;
      }
    }

    int tiles_played_score = 0;
    for (int i = 0; i < RACK_SIZE; ++i)
      tiles_played_score += descending_tile_scores[i] * descending_effective_letter_multipliers[i];
    const int bingo = (tiles_played == RACK_SIZE) ? 50 : 0;
    const int score = tiles_played_score +
                      shadow_mainword_restricted_score * shadow_word_multiplier +
                      shadow_perpendicular_additional_score + bingo;
    if (word_length >= 2) maybe_update_anchor(tiles_played, word_length, current_left_col, score);
    if (tiles_played > max_tiles_to_play) max_tiles_to_play = tiles_played;
  }

  // Whether to record: several tiles, or one tile that no other pass owns.
  static bool nonempty_and_nondup(int tiles_played, bool is_unique) {
    return (tiles_played > 1) || ((tiles_played == 1) && is_unique);
  }

  void shadow_play_right(bool is_unique) {
    const int orig_main_restricted = shadow_mainword_restricted_score;
    const int orig_perp = shadow_perpendicular_additional_score;
    const int orig_wordmul = shadow_word_multiplier;
    const uint32_t orig_rack_cross_set = rack_cross_set;
    bool restricted_any = false;
    const int orig_num_unrestricted = num_unrestricted_multipliers;
    bool changed_multipliers = false;
    const int original_right_col = current_right_col;
    const int original_tiles_played = tiles_played;
    playthrough_blocks_copy = playthrough_blocks;
    num_tiles_played_through_copy = num_tiles_played_through;
    playthrough_bit_rack_copy = playthrough_bit_rack;

    while (current_right_col < (BOARD_SIZE - 1) && tiles_played < number_of_letters_on_rack) {
      ++current_right_col;
      ++tiles_played;
      const uint32_t cross_set = lane.cross_set[current_right_col];
      const uint32_t current_leftx = lane.left_ext[current_right_col];
      if ((cross_set & current_leftx) == 0) break;
      const uint32_t possible =
        cross_set & rack_cross_set & anchor_right_extension_set & current_leftx;
      anchor_right_extension_set = kTrivialCrossSet;
      if (possible == 0) break;

      const int letter_mult = lane.letter_mult[current_right_col];
      const int this_word_mult = lane.word_mult[current_right_col];
      shadow_perpendicular_additional_score += lane.cross_score[current_right_col] * this_word_mult;
      shadow_word_multiplier *= this_word_mult;

      if (std::has_single_bit(possible)) {
        if (!restricted_any) {
          player_rack_shadow_right_copy = rack;
          descending_tile_scores_copy = descending_tile_scores;
          restricted_any = true;
        }
        restrict_tile_and_accumulate_score(possible, letter_mult, this_word_mult,
                                           current_right_col);
      } else {
        if (!changed_multipliers) {
          desc_xw_muls_copy = descending_cross_word_multipliers;
          desc_eff_letter_muls_copy = descending_effective_letter_multipliers;
          changed_multipliers = true;
        }
        insert_unrestricted_multipliers(current_right_col);
      }
      if (cross_set == kTrivialCrossSet) is_unique = true;

      bool found_playthrough = false;
      while (current_right_col + 1 < BOARD_SIZE && !lane.empty[current_right_col + 1]) {
        found_playthrough = true;
        const int8_t ml = lane.letter[current_right_col + 1];
        shadow_mainword_restricted_score += TILE_VALUES[ml];
        playthrough_bit_rack.add_letter(ml);
        ++num_tiles_played_through;
        ++current_right_col;
      }
      if (found_playthrough) ++playthrough_blocks;

      if (nonempty_and_nondup(tiles_played, is_unique)) {
        maybe_recalculate_effective_multipliers();
        shadow_record();
      }
    }

    shadow_mainword_restricted_score = orig_main_restricted;
    shadow_perpendicular_additional_score = orig_perp;
    shadow_word_multiplier = orig_wordmul;
    if (restricted_any) {
      rack = player_rack_shadow_right_copy;
      rack_cross_set = orig_rack_cross_set;
      descending_tile_scores = descending_tile_scores_copy;
    }
    if (changed_multipliers) {
      num_unrestricted_multipliers = orig_num_unrestricted;
      descending_cross_word_multipliers = desc_xw_muls_copy;
      descending_effective_letter_multipliers = desc_eff_letter_muls_copy;
    }
    current_right_col = original_right_col;
    tiles_played = original_tiles_played;
    playthrough_blocks = playthrough_blocks_copy;
    num_tiles_played_through = num_tiles_played_through_copy;
    playthrough_bit_rack = playthrough_bit_rack_copy;
    maybe_recalculate_effective_multipliers();
  }

  void nonplaythrough_shadow_play_left(bool is_unique) {
    for (;;) {
      if ((anchor_right_extension_set & rack_cross_set) != 0) shadow_play_right(is_unique);
      anchor_right_extension_set = kTrivialCrossSet;
      if (current_left_col == 0 || current_left_col == last_anchor_col + 1 ||
          tiles_played >= number_of_letters_on_rack)
        return;
      const uint32_t possible_left = anchor_left_extension_set & rack_cross_set;
      if (possible_left == 0) return;
      anchor_left_extension_set = kTrivialCrossSet;

      --current_left_col;
      ++tiles_played;
      const int letter_mult = lane.letter_mult[current_left_col];
      const int this_word_mult = lane.word_mult[current_left_col];
      shadow_word_multiplier *= this_word_mult;
      if (!try_restrict_tile(possible_left, letter_mult, this_word_mult, current_left_col))
        insert_unrestricted_multipliers(current_left_col);
      shadow_record();
    }
  }

  void playthrough_shadow_play_left(bool is_unique) {
    for (;;) {
      if ((anchor_right_extension_set & rack_cross_set) != 0) shadow_play_right(is_unique);
      anchor_right_extension_set = kTrivialCrossSet;

      uint32_t possible_left = anchor_left_extension_set & rack_cross_set;
      const uint32_t leftx = anchor_left_extension_set;
      anchor_left_extension_set = kTrivialCrossSet;

      if (current_left_col == 0 || current_left_col == last_anchor_col + 1 ||
          tiles_played >= number_of_letters_on_rack)
        break;
      if (possible_left == 0) break;
      --current_left_col;
      ++tiles_played;
      const uint32_t cross_set = lane.cross_set[current_left_col];
      if ((cross_set & leftx) == 0) break;
      possible_left &= cross_set;
      if (possible_left == 0) break;

      const int letter_mult = lane.letter_mult[current_left_col];
      const int this_word_mult = lane.word_mult[current_left_col];
      shadow_perpendicular_additional_score += lane.cross_score[current_left_col] * this_word_mult;
      shadow_word_multiplier *= this_word_mult;
      if (!try_restrict_tile(possible_left, letter_mult, this_word_mult, current_left_col))
        insert_unrestricted_multipliers(current_left_col);
      if (cross_set == kTrivialCrossSet) is_unique = true;
      if (nonempty_and_nondup(tiles_played, is_unique)) shadow_record();
    }
  }

  void shadow_start_nonplaythrough() {
    const uint32_t cross_set = lane.cross_set[current_left_col];
    const uint32_t possible = cross_set & rack_cross_set;
    if (possible == 0) return;
    const int letter_mult = lane.letter_mult[current_left_col];
    const int this_word_mult = lane.word_mult[current_left_col];
    shadow_perpendicular_additional_score = lane.cross_score[current_left_col] * this_word_mult;
    // 0 for the one-tile record: a lone tile makes no main word, only the
    // cross-word already counted in the perpendicular score.
    shadow_word_multiplier = 0;
    if (!try_restrict_tile(possible, letter_mult, this_word_mult, current_left_col))
      insert_unrestricted_multipliers(current_left_col);
    ++tiles_played;
    // A single tile in the transposed pass duplicates the horizontal pass's
    // play unless no cross-word forms at its square (cf.
    // single_tile_duplicates_horizontal_pass).
    const bool is_unique = !transposed || cross_set == kTrivialCrossSet;
    if (is_unique) shadow_record();
    shadow_word_multiplier = this_word_mult;
    maybe_recalculate_effective_multipliers();
    nonplaythrough_shadow_play_left(is_unique);
  }

  void shadow_start_playthrough(int current_letter) {
    for (;;) {
      shadow_mainword_restricted_score += TILE_VALUES[current_letter];
      playthrough_bit_rack.add_letter(current_letter);
      ++num_tiles_played_through;
      if (current_left_col == 0 || current_left_col == last_anchor_col + 1) break;
      --current_left_col;
      if (lane.empty[current_left_col]) {
        ++current_left_col;
        break;
      }
      current_letter = lane.letter[current_left_col];
    }
    ++playthrough_blocks;
    playthrough_shadow_play_left(!transposed);
  }

  void shadow_start() {
    if ((anchor_left_extension_set | anchor_right_extension_set) == 0) return;
    const uint32_t original_rack_cross_set = rack_cross_set;
    full_rack = rack;
    if (lane.empty[current_left_col])
      shadow_start_nonplaythrough();
    else
      shadow_start_playthrough(lane.letter[current_left_col]);
    rack_cross_set = original_rack_cross_set;
    rack = full_rack;
  }

  // Runs the walk for one anchor. Returns whether any slot was recorded.
  bool shadow_play_for_anchor(int col) {
    current_left_col = col;
    current_right_col = col;
    anchor_left_extension_set = lane.left_ext[col];
    anchor_right_extension_set = lane.right_ext[col];
    num_unrestricted_multipliers = 0;
    descending_effective_letter_multipliers.fill(0);
    last_word_multiplier = 1;
    descending_tile_scores = full_rack_descending;
    shadow_mainword_restricted_score = 0;
    shadow_perpendicular_additional_score = 0;
    shadow_word_multiplier = 1;
    current_anchor_col = col;
    tiles_played = 0;
    max_tiles_to_play = 0;
    playthrough_blocks = 0;
    num_tiles_played_through = 0;
    playthrough_bit_rack = BitRack{};
    reset_anchors();
    shadow_start();
    return max_tiles_to_play > 0;
  }
};

}  // namespace

std::vector<ShadowExtent> ShadowMoveGen::extents(
  const Rack& rack, const WordMap* wm,
  const std::array<bool, kMaxPlayTiles + 1>* nonplaythrough_has_word) const {
  board_.ensure_movegen_caches(dict_);
  const TileCounts& counts = rack.counts();
  const int rack_tiles = counts.size();

  // MAGPIE's nonplaythrough_has_word_of_length. Callers that price leaves have
  // usually made these lookups already and pass the result in.
  std::array<bool, kMaxPlayTiles + 1> has_word_storage{};
  if (nonplaythrough_has_word == nullptr && wm != nullptr) {
    WmpSubracks subracks;
    int rt = 0;
    wmp_rack_subracks(rack, subracks, rt);
    for (int size = 2; size <= rack_tiles && size <= kMaxPlayTiles; ++size) {
      for (const BitRack& sub : subracks[size]) {
        if (wm->lookup(size, sub).count > 0) {
          has_word_storage[size] = true;
          break;
        }
      }
    }
    nonplaythrough_has_word = &has_word_storage;
  }
  const std::array<bool, kMaxPlayTiles + 1> empty_has_word{};
  const std::array<bool, kMaxPlayTiles + 1>& has_word =
    nonplaythrough_has_word != nullptr ? *nonplaythrough_has_word : empty_has_word;

  std::vector<ShadowExtent> out;
  for (int orient = 0; orient < 2; ++orient) {
    const bool transposed = (orient == 1);
    View view{board_, transposed};
    const CrossChecks& cross = board_.cross_checks(transposed);
    const Anchors& anchors = board_.gaddag_anchors(transposed);
    for (int r = 0; r < BOARD_SIZE; ++r) {
      bool any = false;
      for (int c = 0; c < BOARD_SIZE && !any; ++c) any = anchors[idx(r, c)];
      if (!any) continue;
      const ShadowLane lane = build_shadow_lane(view, cross, anchors, dict_, r);
      ShadowGen gen{lane, transposed, counts, wm, has_word};
      gen.last_anchor_col = BOARD_SIZE;  // no previous anchor in this row
      for (int c = 0; c < BOARD_SIZE; ++c) {
        if (!lane.anchor[c]) continue;
        if (gen.shadow_play_for_anchor(c)) {
          for (int i = 0; i < gen.num_touched; ++i) {
            const ShadowGen::Slot& s = gen.slots[gen.touched[i]];
            if (s.tiles_to_play == 0) continue;
            ShadowExtent e;
            e.transposed = transposed;
            e.row = r;
            e.length = s.word_length;
            e.placed = s.tiles_to_play;
            e.pt = set_playthrough_bitrack(lane, s.rightmost, s.playthrough_blocks);
            e.score_bound = s.score;
            e.leftmost_start_col = s.leftmost;
            e.rightmost_start_col = s.rightmost;
            out.push_back(e);
          }
        }
        // As in MAGPIE, an occupied anchor pushes the left limit one further
        // right: a play covering the next square also covers this tile, so it was
        // already bounded from this anchor.
        gen.last_anchor_col = c;
        if (!lane.empty[c]) ++gen.last_anchor_col;
      }
    }
  }
  return out;
}

}  // namespace scribblez
