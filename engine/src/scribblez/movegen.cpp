#include "scribblez/movegen.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

constexpr uint32_t ALL_LETTERS_MASK = (1u << 26) - 1u;

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

struct CrossCheck {
  uint32_t mask = ALL_LETTERS_MASK;
  int score = 0;  // sum of TILE_VALUES of existing perpendicular letters (blanks = 0)
  bool has_neighbor = false;
};

// Per-square scratch for one generate() pass -- stack-allocated, so a turn's
// move generation does no heap allocation for these.
using CrossChecks = std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>;
using Anchors = std::array<bool, BOARD_SIZE * BOARD_SIZE>;

constexpr int idx(int r, int c) { return r * BOARD_SIZE + c; }

// Compute cross-checks for one orientation. cross[r*15+c] applies to placing a
// tile at view position (r, c); the perpendicular run is along increasing/decreasing r at fixed c.
CrossChecks compute_cross_checks(const View& view, const Dictionary& dict) {
  CrossChecks out{};
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph here = view.at(r, c);
      if (!here.is_empty()) continue;  // cross-check only meaningful for empty squares

      // Walk up (decreasing r at fixed c) collecting existing letters (top-to-bottom order: from
      // `top` row to r-1).
      int top = r - 1;
      while (top >= 0 && !view.at(top, c).is_empty()) --top;
      ++top;  // top is now first existing row above r (or r if none)

      // Walk down (increasing r at fixed c).
      int bot = r + 1;
      while (bot < BOARD_SIZE && !view.at(bot, c).is_empty()) ++bot;
      --bot;  // bot is last existing row below r (or r if none)

      CrossCheck cc;
      cc.has_neighbor = (top < r) || (bot > r);
      if (!cc.has_neighbor) {
        cc.mask = ALL_LETTERS_MASK;
        cc.score = 0;
      } else {
        // Walk the dictionary through the existing prefix (top..r-1).
        uint32_t prefix_node = dict.root();
        int prefix_score = 0;
        bool prefix_ok = true;
        for (int rr = top; rr <= r - 1; ++rr) {
          Glyph sq = view.at(rr, c);
          auto tr = dict.step(prefix_node, sq.letter());
          if (!tr.valid) {
            prefix_ok = false;
            break;
          }
          prefix_node = tr.next;
          if (!sq.is_blank()) prefix_score += TILE_VALUES[sq.letter()];
        }
        // Sum suffix score upfront.
        int suffix_score = 0;
        for (int rr = r + 1; rr <= bot; ++rr) {
          Glyph sq = view.at(rr, c);
          if (!sq.is_blank()) suffix_score += TILE_VALUES[sq.letter()];
        }
        cc.score = prefix_score + suffix_score;
        uint32_t mask = 0;
        if (prefix_ok) {
          for (Tile L = Tile::of(0); L < 26; ++L) {
            auto tr_l = dict.step(prefix_node, L);
            if (!tr_l.valid) continue;
            bool acc = tr_l.accepts;
            uint32_t n = tr_l.next;
            bool ok = true;
            for (int rr = r + 1; rr <= bot; ++rr) {
              auto tr_s = dict.step(n, view.at(rr, c).letter());
              if (!tr_s.valid) {
                ok = false;
                break;
              }
              acc = tr_s.accepts;
              n = tr_s.next;
            }
            if (ok && acc) mask |= (1u << L);
          }
        }
        cc.mask = mask;
      }
      out[idx(r, c)] = cc;
    }
  }
  return out;
}

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
// view columns [start_col, end_col_excl) on `row`. Squares already filled on the
// board contribute their existing letters; empty squares in the span are newly
// placed and their (letter, is_blank) come from `placed_letter`/`placed_blank`.
// Scoring (premiums, cross-words, bingo) is shared by both generators so the
// two algorithms produce byte-identical moves.
Move build_play(const View& view, const CrossChecks& cross, int row, int start_col,
                int end_col_excl, const std::array<Tile, BOARD_SIZE>& placed_letter,
                const std::array<bool, BOARD_SIZE>& placed_blank) {
  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = !view.transposed;
  auto start_bc = view.to_board(row, start_col);
  m.start_row = start_bc.first;
  m.start_col = start_bc.second;
  m.square_mask = 0;

  int n_placed = 0;
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
      m.glyphs[n_placed++] = Glyph::played(L, is_blank);  // in word order
      m.square_mask |= static_cast<uint16_t>(1u << (c - start_col));
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

  m.score = main_letter_sum * word_mult + cross_total;
  if (n_placed == RACK_SIZE) m.score += 50;  // bingo
  return m;
}

// Anchors for the GADDAG (Gordon) generator. Unlike the DAWG generator's
// anchors, these are direction-specific (computed in view coordinates, where
// generation runs along increasing column) and include occupied squares. This
// matches Macondo's `updateAnchors` and is required for correctness: it ensures
// every generated word is anchored at its rightmost square, so the rightward
// boundary (a possible trailing existing tile) is never silently ignored.
//   - An occupied square is an anchor iff it has no tile immediately to its
//     right (it is the rightmost square of an existing run).
//   - An empty square is an anchor iff both its left and right neighbors are
//     empty and it has a tile directly above or below (a pure cross-hook).
// The empty board is special-cased to a single anchor at the center.
Anchors compute_gaddag_anchors(const View& view) {
  Anchors anchor{};
  bool any_tile = false;
  for (int r = 0; r < BOARD_SIZE && !any_tile; ++r)
    for (int c = 0; c < BOARD_SIZE && !any_tile; ++c)
      if (!view.at(r, c).is_empty()) any_tile = true;
  if (!any_tile) {
    anchor[idx(CENTER, CENTER)] = true;
    return anchor;
  }
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const bool here = !view.at(r, c).is_empty();
      const bool tile_left = c > 0 && !view.at(r, c - 1).is_empty();
      const bool tile_right = c < BOARD_SIZE - 1 && !view.at(r, c + 1).is_empty();
      const bool tile_above = r > 0 && !view.at(r - 1, c).is_empty();
      const bool tile_below = r < BOARD_SIZE - 1 && !view.at(r + 1, c).is_empty();
      if (here) {
        if (!tile_right) anchor[idx(r, c)] = true;
      } else if (!tile_left && !tile_right && (tile_above || tile_below)) {
        anchor[idx(r, c)] = true;
      }
    }
  }
  return anchor;
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
    if (accepts_here && col > current_anchor_col && total_placed > 0) {
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
};

}  // namespace

MoveGenerator::MoveGenerator(const Board& board, const Dictionary& dict)
    : board_(board), dict_(dict) {}

std::vector<Move> MoveGenerator::generate(const Rack& rack, GenAlgo algo) {
  std::vector<Move> out;
  for (int orient = 0; orient < 2; ++orient) {
    bool transposed = (orient == 1);
    View view{board_, transposed};
    auto cross = compute_cross_checks(view, dict_);
    const auto before = out.size();
    if (algo == GenAlgo::GADDAG) {
      auto anchors = compute_gaddag_anchors(view);
      GaddagGen st{view, dict_, cross, anchors, rack.counts(), out};
      for (int r = 0; r < BOARD_SIZE; ++r) st.generate_for_row(r);
    } else {
      auto anchors = compute_anchors(view);
      GenState st{view, dict_, cross, anchors, rack.counts(), out};
      for (int r = 0; r < BOARD_SIZE; ++r) st.generate_for_row(r);
    }
    // Single-tile plays are produced by both orientations; treat horizontal as
    // canonical and suppress them from the transposed (vertical) pass.
    if (transposed) {
      out.erase(std::remove_if(out.begin() + before, out.end(),
                               [](const Move& m) { return m.num_glyphs() == 1; }),
                out.end());
    }
  }
  return out;
}

}  // namespace scribblez
