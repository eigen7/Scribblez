#include "scribblez/movegen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// A View presents the board with optional transpose, so the same generator can
// produce both horizontal and vertical plays.
struct View {
  const Board& board;
  bool transposed;
  Glyph at(int r, int c) const { return transposed ? board.at(c, r) : board.at(r, c); }
  Premium premium_at(int r, int c) const {
    return transposed ? board.premium_at(c, r) : board.premium_at(r, c);
  }
  // Translate view coords back to board coords.
  std::pair<int, int> to_board(int r, int c) const {
    return transposed ? std::make_pair(c, r) : std::make_pair(r, c);
  }
};

// Per-square scratch for one generate() pass. CrossCheck and the cross-check
// values themselves live on the Board (persisted and incrementally maintained);
// see board.h / board.cpp.
using CrossChecks = std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>;
using Anchors = std::array<bool, BOARD_SIZE * BOARD_SIZE>;

constexpr int idx(int r, int c) { return r * BOARD_SIZE + c; }

// Compute anchor squares for this view. An anchor is an empty square adjacent
// (in any of the 4 directions) to a filled square. Special case: if the board
// is completely empty, the single anchor is the center square.
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
  static const int dr[4] = {-1, 1, 0, 0};
  static const int dc[4] = {0, 0, -1, 1};
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!view.at(r, c).is_empty()) continue;
      for (int k = 0; k < 4; ++k) {
        int nr = r + dr[k], nc = c + dc[k];
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

// Build a PLAY Move (placed tiles, main word, score) for the run that occupies
// view columns [start_col, end_col_excl) on `row`. Squares already filled on
// the board contribute their existing letters; empty squares in the span are
// newly placed and their (letter, is_blank) come from
// `placed_letter`/`placed_blank`. Scoring (premiums, cross-words, bingo) is
// shared by both generators so the two algorithms produce byte-identical moves.
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
      played[n_placed++] = Glyph::played(L, is_blank);  // in word order
      square_mask |= static_cast<uint16_t>(1u << c);    // absolute lane index
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
  return Move::play(!view.transposed, row, square_mask, static_cast<uint16_t>(score), played.data(),
                    n_placed);
}

struct GenState {
  GenState(const View& view, const Dictionary& dict, const CrossChecks& cross,
           const Anchors& anchor, TileCounts rack, std::vector<Move>& out)
      : view(view), dict(dict), cross(cross), anchor(anchor), rack(std::move(rack)), out(out) {}

  const View& view;
  const Dictionary& dict;
  const CrossChecks& cross;
  const Anchors& anchor;
  TileCounts rack;  // available-tile scratch (built from the player's rack)
  std::vector<Move>& out;

  // Recursion state for the current word being built.
  std::vector<std::pair<Tile, bool>> left_letters;  // (letter, is_blank) in left-to-right order
  struct RightTile {
    int col;
    Tile letter;
    bool is_blank;
  };
  std::vector<RightTile> right_placed;
  int current_row = 0;
  int current_anchor_col = 0;
  int case_b_start_col = -1;  // for case B: where the existing prefix begins; -1 means case A

  void emit_move(int start_col, int end_col_excl);

  void extend_right(int col, uint32_t node, bool accepts_here);
  void left_part(int limit, uint32_t node);
  void generate_for_row(int row);
};

void GenState::emit_move(int start_col, int end_col_excl) {
  // Lay the recursion's placed tiles into a per-column strip, then defer to the
  // shared scorer.
  std::array<Tile, BOARD_SIZE> placed_letter{};
  std::array<bool, BOARD_SIZE> placed_blank{};
  int li = 0, ri = 0;
  for (int c = start_col; c < end_col_excl; ++c) {
    if (!view.at(current_row, c).is_empty()) continue;  // existing tile
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
    // Single-tile plays are produced by both orientations; treat horizontal as
    // canonical and never emit them from the transposed (vertical) pass.
    bool suppress = view.transposed && total_placed == 1;
    if (accepts_here && col > current_anchor_col && total_placed > 0 && !suppress) {
      int start_col;
      if (case_b_start_col >= 0) {
        start_col = case_b_start_col;
      } else {
        start_col = current_anchor_col - (int)left_letters.size();
      }
      emit_move(start_col, col);
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
  // First: attempt to stop the left part here and extend right from anchor.
  // `accepts_here` is irrelevant: extend_right at col == anchor_col cannot emit
  // (the emit guard requires col > anchor_col), so any value is safe.
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

    // Decide between case A (no immediate-left filled tile) and case B (immediate-left filled).
    if (col > 0 && !view.at(row, col - 1).is_empty()) {
      // Case B: walk left through existing tiles to find prefix start.
      int start_c = col - 1;
      while (start_c - 1 >= 0 && !view.at(row, start_c - 1).is_empty()) --start_c;
      // Traverse the dictionary through the existing prefix.
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
      // Case A: compute left limit.
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
// GADDAG generator (Gordon's algorithm).
//
// This is the move generator described in the design doc. The GADDAG spine
// lets a single left-to-right scan from each anchor place tiles leftward
// (following reversed-prefix arcs), then "shift direction" through the
// separator token to place tiles rightward. Cross-checks (computed from the
// forward DAWG, identical to the reference generator) gate perpendicular-word
// validity, and build_play() does the scoring. Single-tile plays are only
// emitted from the horizontal pass to avoid duplicates.
// ---------------------------------------------------------------------------
struct GaddagGen {
  GaddagGen(const View& view, const Dictionary& dict, const CrossChecks& cross,
            const Anchors& anchor, TileCounts rack, std::vector<Move>& out)
      : view(view), dict(dict), cross(cross), anchor(anchor), rack(std::move(rack)), out(out) {}

  const View& view;
  const Dictionary& dict;
  const CrossChecks& cross;
  const Anchors& anchor;
  TileCounts rack;  // available-tile scratch (built from the player's rack)
  std::vector<Move>& out;

  int current_row = 0;
  int current_anchor_col = 0;
  int last_anchor_col = 100;  // sentinel: no previous anchor this row
  int tiles_played = 0;
  std::array<Tile, BOARD_SIZE> strip_letter{};
  std::array<bool, BOARD_SIZE> strip_blank{};
  // The current row's cells, flattened once per row so the recursion indexes a
  // plain array instead of going through View::at()'s transpose branch.
  std::array<Glyph, BOARD_SIZE> row_cells{};

  void record(int leftstrip, int rightstrip) {
    // Single-tile plays are produced by both orientations; treat horizontal as
    // canonical and never emit them from the transposed (vertical) pass.
    if (view.transposed && tiles_played == 1) return;
    out.push_back(
      build_play(view, cross, current_row, leftstrip, rightstrip + 1, strip_letter, strip_blank));
  }

  // Gordon's GoOn: we have just transitioned to `new_node` by placing/using
  // letter L at `col`; `accepts` says the path so far spells a complete word.
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
      // Keep extending to the left (but never past the previous anchor).
      if (col > 0 && col - 1 != last_anchor_col) {
        recursive_gen(col - 1, new_node, leftstrip, rightstrip);
      }
      // Shift direction through the separator to extend right of the anchor.
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
    // Iterate this node's child arcs once (KWG arcs are sorted by tile value, so
    // letters come out A..Z) rather than scanning the arc list 26 times.
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

  // Generate exactly the plays canonically anchored at (row, col). `last_anchor`
  // is the nearest anchor to the left in this row (100 if none), matching the
  // left-extension bound generate_for_row applies, so this produces the same
  // moves with the same dedup regardless of processing order.
  void generate_one_anchor(int row, int col, int last_anchor) {
    current_row = row;
    for (int c = 0; c < BOARD_SIZE; ++c) row_cells[c] = view.at(row, c);
    last_anchor_col = last_anchor;
    current_anchor_col = col;
    tiles_played = 0;
    recursive_gen(col, dict.gaddag_root(), col, col);
  }
};

// The player's tile values (blanks count as 0) sorted descending -- the input
// to the shadow score bound's "best tiles in best multipliers" estimate.
std::vector<int> rack_values_desc(const TileCounts& rack) {
  std::vector<int> v;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    for (int i = 0; i < rack.count(L); ++i) v.push_back(TILE_VALUES[L]);
  }
  for (int i = 0; i < rack.blanks(); ++i) v.push_back(0);
  std::sort(v.begin(), v.end(), std::greater<int>());
  return v;
}

// Per-lane (one view row) scratch for the shadow score bound, computed once and
// reused across the row's anchors. For empty squares it records the premium
// multipliers, cross-word score/neighbor, whether any rack tile can be played
// there, and the highest-value rack tile the cross-check permits.
struct LaneInfo {
  std::array<bool, BOARD_SIZE> filled{}, placeable{};
  std::array<int, BOARD_SIZE> tval{}, lmul{}, wmul{}, cscore{}, maxval{};
  std::array<bool, BOARD_SIZE> cneigh{};
  std::array<int, BOARD_SIZE + 1> pref{};  // prefix sum of filled tile values
};

void build_lane(const View& view, const CrossChecks& cross, int row, uint32_t rack_letter_mask,
                bool has_blank, LaneInfo& lane) {
  lane = LaneInfo{};
  for (int c = 0; c < BOARD_SIZE; ++c) {
    const Glyph g = view.at(row, c);
    if (!g.is_empty()) {
      lane.filled[c] = true;
      lane.tval[c] = g.is_blank() ? 0 : TILE_VALUES[g.letter()];
      continue;
    }
    const Premium p = view.premium_at(row, c);
    lane.lmul[c] = p.letter_mult();
    lane.wmul[c] = p.word_mult();
    const CrossCheck& cc = cross[idx(row, c)];
    lane.cscore[c] = cc.score;
    lane.cneigh[c] = cc.has_neighbor;
    // Highest-value rack tile this square's cross-check permits (a blank can
    // satisfy any non-empty mask, scoring 0). A square no rack tile can fill
    // makes any covering window infeasible.
    const uint32_t allowed = cc.mask & rack_letter_mask;
    if (allowed != 0) {
      lane.placeable[c] = true;
      int mv = 0;
      for (Tile L = Tile::of(0); L < 26; ++L) {
        if ((allowed & (1u << L)) && TILE_VALUES[L] > mv) mv = TILE_VALUES[L];
      }
      lane.maxval[c] = mv;
    } else if (has_blank && cc.mask != 0) {
      lane.placeable[c] = true;
      lane.maxval[c] = 0;
    }
  }
  for (int c = 0; c < BOARD_SIZE; ++c) lane.pref[c + 1] = lane.pref[c] + lane.tval[c];
}

// Fill `out[e]` with an admissible upper bound on the raw score of a play that
// places exactly e tiles canonically anchored at `col` (or -1 if none). It
// enumerates every contiguous window [a, b] covering the anchor (a no further
// left than the previous anchor), and for each window placing 1..rack_size tiles
// bounds the score by the smaller of two over-estimates of the placed
// contribution -- greedily pairing the top rack tiles with the window's best
// letter multipliers (count-tight), and summing each square's max permitted tile
// (cross-check-tight) -- times the product of word multipliers, plus generous
// cross-word and bingo terms. Every term over-estimates, so the bound never
// underestimates a real play's score, which makes best-first pruning exact.
void anchor_score_bounds(const LaneInfo& lane, int col, int last_anchor_col,
                         const std::vector<int>& rack_vals_desc,
                         std::array<int, kMaxPlayTiles + 1>& out) {
  out.fill(-1);
  const int rack_size = static_cast<int>(rack_vals_desc.size());
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
      psm += lane.maxval[c] * lm;  // per-square max placed contribution
      if (lane.cneigh[c]) cross_sum += (lane.cscore[c] + lane.maxval[c] * lm) * lane.wmul[c];
    };
    for (int c = a; c < col; ++c) add_square(c);
    for (int b = col; b < BOARD_SIZE; ++b) {
      add_square(b);
      if (e == 0) continue;
      if (e > e_cap) break;  // e only grows with b
      if (bad > 0) break;    // window has an unfillable square (stays so as b grows)
      int wl = a;
      while (wl - 1 >= 0 && lane.filled[wl - 1]) --wl;
      int wr = b;
      while (wr + 1 < BOARD_SIZE && lane.filled[wr + 1]) ++wr;
      const int existing = lane.pref[wr + 1] - lane.pref[wl];
      // Count-tight estimate: top-e rack values paired with the window's highest
      // letter multipliers (3s, then 2s, then 1s).
      int greedy = 0, taken = 0;
      for (int j = 0; j < c3 && taken < e; ++j) greedy += rack_vals_desc[taken++] * 3;
      for (int j = 0; j < c2 && taken < e; ++j) greedy += rack_vals_desc[taken++] * 2;
      while (taken < e) greedy += rack_vals_desc[taken++];
      const int placed = std::min(greedy, psm);
      const long main_word = static_cast<long>(placed + existing) * wprod;
      const long sc = main_word + cross_sum + (e == RACK_SIZE ? 50 : 0);
      if (sc > out[e]) out[e] = static_cast<int>(sc);
    }
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

  std::vector<ShadowAnchor> out;
  LaneInfo lane;
  for (int orient = 0; orient < 2; ++orient) {
    const bool transposed = (orient == 1);
    View view{board_, transposed};
    const CrossChecks& cross = board_.cross_checks(transposed);
    const Anchors& anchors = board_.gaddag_anchors(transposed);
    for (int r = 0; r < BOARD_SIZE; ++r) {
      // Lanes with no anchors contribute nothing; skip the lane build.
      bool any = false;
      for (int c = 0; c < BOARD_SIZE && !any; ++c) any = anchors[idx(r, c)];
      if (!any) continue;
      build_lane(view, cross, r, rack_letter_mask, has_blank, lane);
      int prev = -1;  // nearest anchor to the left in this row
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

}  // namespace scribblez
