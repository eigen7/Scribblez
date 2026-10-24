// The hasty-rollout kernel: one thread block per rollout. Each ply enumerates
// every word span of the board, probes the word map (or, for a subrack with
// blanks, the blank table) for every (span, subrack) item, scores each play
// it finds, and reduces to HastyBot's argmax under hasty_move_better's total
// order; then applies the play, redraws, and recomputes the cross-checks the
// play changed. It replicates movegen.cpp's WordMap generator
// (wmp_emit_leftmost_anchor / wmp_try_word / build_play), HastyEquity::equity
// and Board::cross_check_at; where a line here has a counterpart there, the
// names match. With tiles in the bag, a ply runs in two passes and skips the
// spans and items whose equity bound cannot reach the best move found so far
// (see "The span bound"); the bound is exact, so the moves are unchanged.
//
// Per-block state lives in shared memory, except the ply's span list, which
// goes to a per-block slot of global scratch (L2-resident while hot): one
// pass over all 30 lanes per ply, so the items of a whole ply share one
// work queue and the slow ones (a blank hit returns dozens of words) stop
// stalling the block at a chunk barrier.

#include "rollout_kernel.h"

#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace scribblez::rollout_gpu {

using namespace rollout_abi;

namespace {

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
  }
}

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMinBlocksPerSm = 4;
// Each lane (a row of either view) has at most 15 leftmost anchors x 15
// right ends with at least two squares per word, which caps it at 120.
constexpr int kLanes = 2 * kBoardSize;
constexpr int kMaxSpans = kLanes * 120;
// Scratch slots: more than the blocks a device can keep resident.
constexpr int kScratchSlots = 512;
constexpr uint32_t kAllLetters = (1u << 26) - 1u;
constexpr uint32_t kArcMask = 0x003fffffu;
constexpr uint32_t kIsEndBit = 0x00400000u;
constexpr uint32_t kAcceptsBit = 0x00800000u;
constexpr int kMaxBlanks = 2;

// A word span of one lane: the play covers view-row `row`, columns [wl, wr],
// placing `placed` tiles on the columns of `placed_mask`; `pt` is the
// multiset of the tiles it plays through and `admitted` the union of its
// placed squares' cross-checks.
struct Span {
  uint64_t pt_lo, pt_hi;
  uint32_t admitted;
  uint16_t placed_mask;
  uint16_t packed;  // view:1 row:4 wl:4 wr:4 placed:3
};
static_assert(sizeof(Span) == 24);

__device__ inline int span_view(const Span& s) { return s.packed & 1; }
__device__ inline int span_row(const Span& s) { return (s.packed >> 1) & 15; }
__device__ inline int span_wl(const Span& s) { return (s.packed >> 5) & 15; }
__device__ inline int span_wr(const Span& s) { return (s.packed >> 9) & 15; }
__device__ inline int span_placed(const Span& s) { return (s.packed >> 13) & 7; }

// A span's score bound parts (span_parts): the fixed score (playthrough sum
// times the word multiplier, the cross words' fixed scores, the bingo) and
// the placed squares' coefficients, highest first, packed a byte each.
struct SpanParts {
  int fixed;
  uint64_t coef;
};

// A span with its score bound parts and equity bound, as bound_spans
// partitions them.
struct BoundedSpan {
  Span span;
  SpanParts parts;
  double bound;
};

// One block's global scratch: the ply's spans, their score bound parts, and
// the two item-space offsets over them (blank-free subracks, subracks with
// blanks); then each span's equity bound, and the spans partitioned by it:
// the first pass's, then the rest (see bound_spans).
struct Scratch {
  Span spans[kMaxSpans];
  SpanParts parts[kMaxSpans];
  uint32_t off[kMaxSpans + 1];
  uint32_t offb[kMaxSpans + 1];
  double bound[kMaxSpans];
  BoundedSpan parted[kMaxSpans];
};

struct Tables {
  const uint64_t* slots;
  uint32_t slot_offset[kMaxWordLen + 1];
  uint64_t slot_mask[kMaxWordLen + 1];
  const uint4* words16;
  uint32_t word_base[kMaxWordLen + 1];
  const uint32_t* leave_keys;
  const float* leave_values;
  uint32_t leave_mask;
  int leave_shift;
  const uint32_t* arcs;
  uint32_t dawg_root;
  const double* peg;  // [kPegTiles][kPegBags]
  const uint64_t* blank_slots[2];
  uint32_t blank_slot_offset[2][kMaxWordLen + 1];
  uint64_t blank_slot_mask[2][kMaxWordLen + 1];
  const uint4* blank_lists[2];
  Scratch* scratch;
  int* scratch_busy;
};

__constant__ Tables c_tables;
__constant__ uint8_t c_letter_mult[kSquares];
__constant__ uint8_t c_word_mult[kSquares];
__constant__ int c_tile_values[26];
__constant__ int c_tile_distribution[kTileKinds];

// Cycle counters per kernel phase, summed over blocks by thread 0, and the
// work counters, summed per warp then over warps.
__device__ unsigned long long g_phase_cycles[kPhases];
__device__ unsigned long long g_counts[kCounts];

struct WorkCounts {
  unsigned items = 0, hits = 0, words = 0, plays = 0;
};

// A candidate move with its hasty_move_better key and what apply() needs;
// 48 bytes, shuffled across a warp as 12 words.
struct Cand {
  double eq;
  uint64_t t_hi, t_lo;  // Macondo's tile sequence, big-endian bytes
  int16_t score;
  uint8_t valid, type;  // type: 0 PLAY, 1 EXCHANGE
  uint8_t row, col, vertical, n, len;
  uint8_t view, vrow, wl;
  uint16_t mask;
  uint8_t glyphs[kRackSize];
  uint8_t pad;
};
static_assert(sizeof(Cand) == 48);

union CandWords {
  Cand c;
  uint32_t w[12];
};

// A word as one 16-byte load.
union Word16 {
  uint4 v;
  uint8_t b[16];
};

struct Smem {
  uint8_t sq[kSquares];
  uint32_t xmask[2][kSquares];
  int16_t xscore[2][kSquares];
  uint8_t xnb[2][kSquares];
  uint8_t rack[2][kTileKinds];
  uint8_t leave[2][kTileKinds];  // each player's rack after its last move, before its draw
  Bag bag;
  int scores[2];
  int num_tiles;
  int game_over;
  int scoreless;  // consecutive passes and exchanges
  int slot;
  // Per ply.
  int mover;
  int rack_size;
  int rack_blanks;
  int other_pv;
  uint32_t rack_letters;  // letters the rack can place (all, with a blank)
  uint8_t sorted_rack[kRackSize];
  float leave_bound[kRackSize + 1];  // per tiles played, the best leave value left
  uint32_t sub_values[128];          // per subrack mask, its face values highest first, 4 bits each
  // Per subrack mask over sorted_rack: the leave's value and point value
  // (the mask's tiles taken), and the subrack's own letters.
  float sub_lv[128];
  int16_t sub_pv[128];
  uint64_t sub_lo[128], sub_hi[128];
  uint32_t sub_letters[128];
  // Subrack masks by (tile count, blank count).
  uint8_t masks[kRackSize + 1][kMaxBlanks + 1][35];
  uint8_t n_masks[kRackSize + 1][kMaxBlanks + 1];
  int nspans;
  int nfront, nrest;  // bound_spans' partition: the first pass's spans, the rest
  double best_bound;  // the highest span bound
  double bar;         // items whose bound falls below it are skipped
  int nitems, nitemsb;
  int next_item, next_itemb;  // the work queues' heads
  int overflow;
  Cand warp_best[kWarps];
  double warp_max[kWarps];
  int warp_sum[kWarps], warp_sumb[kWarps];
  // A warp's 32 blank items in flight: their spans, subracks and hit lists.
  struct WarpItems {
    Span span[32];
    const uint4* list[32];
    uint8_t m[32], nb[32];
  } warp_items[kWarps];
  Cand chosen;
  uint8_t placed_r[kRackSize], placed_c[kRackSize];
};

// ---- Board access in view coordinates ----

__device__ inline int nat_idx(int view, int r, int c) {
  return view ? c * kBoardSize + r : r * kBoardSize + c;
}
__device__ inline uint8_t sq_view(const Smem& s, int view, int r, int c) {
  return s.sq[nat_idx(view, r, c)];
}
__device__ inline int tile_value_of_glyph(uint8_t g) {
  return glyph_is_blank(g) ? 0 : c_tile_values[glyph_letter(g)];
}
__device__ inline bool is_vowel(int letter) {
  return letter == 0 || letter == 4 || letter == 8 || letter == 14 || letter == 20;
}

// ---- The DAWG ----

struct DStep {
  uint32_t next;
  bool valid;
  bool accepts;
};

__device__ inline DStep dict_step(uint32_t node, uint8_t tile_value) {
  if (node == 0) return {0, false, false};
  for (uint32_t i = node;; ++i) {
    const uint32_t a = c_tables.arcs[i];
    if ((a >> 24) == tile_value) return {a & kArcMask, true, (a & kAcceptsBit) != 0};
    if (a & kIsEndBit) return {0, false, false};
  }
}

// Board::cross_check_letter_mask by a warp: lane L walks the run below with
// letter L placed, and a ballot collects the mask.
__device__ uint32_t cross_check_letter_mask(const Smem& s, int view, int c, uint32_t prefix_node,
                                            int r, int bot, int lane) {
  bool fits = false;
  if (lane < 26) {
    const DStep tr = dict_step(prefix_node, uint8_t(lane + 1));
    if (tr.valid) {
      bool acc = tr.accepts;
      uint32_t node = tr.next;
      bool ok = true;
      for (int rr = r + 1; rr <= bot; ++rr) {
        const DStep st = dict_step(node, uint8_t(glyph_letter(sq_view(s, view, rr, c)) + 1));
        if (!st.valid) {
          ok = false;
          break;
        }
        acc = st.accepts;
        node = st.next;
      }
      fits = ok && acc;
    }
  }
  return __ballot_sync(0xffffffffu, fits);
}

// Board::cross_check_at by a warp, written into the view's arrays.
__device__ void recompute_cross_check(Smem& s, int view, int r, int c, int lane) {
  const int i = r * kBoardSize + c;
  if (sq_view(s, view, r, c) != 0) {
    s.xmask[view][i] = kAllLetters;
    s.xscore[view][i] = 0;
    s.xnb[view][i] = 0;
    return;
  }
  int top = r;
  while (top - 1 >= 0 && sq_view(s, view, top - 1, c) != 0) --top;
  int bot = r;
  while (bot + 1 < kBoardSize && sq_view(s, view, bot + 1, c) != 0) ++bot;
  if (top == r && bot == r) {
    s.xmask[view][i] = kAllLetters;
    s.xscore[view][i] = 0;
    s.xnb[view][i] = 0;
    return;
  }
  uint32_t prefix_node = c_tables.dawg_root;
  int score = 0;
  bool prefix_ok = true;
  for (int rr = top; rr <= r - 1; ++rr) {
    const uint8_t g = sq_view(s, view, rr, c);
    const DStep tr = dict_step(prefix_node, uint8_t(glyph_letter(g) + 1));
    if (!tr.valid) {
      prefix_ok = false;
      break;
    }
    prefix_node = tr.next;
    score += tile_value_of_glyph(g);
  }
  for (int rr = r + 1; rr <= bot; ++rr) score += tile_value_of_glyph(sq_view(s, view, rr, c));
  const uint32_t mask =
    prefix_ok ? cross_check_letter_mask(s, view, c, prefix_node, r, bot, lane) : 0;
  if (lane == 0) {
    s.xscore[view][i] = int16_t(score);
    s.xnb[view][i] = 1;
    s.xmask[view][i] = mask;
  }
}

// ---- The tables ----

__device__ inline uint64_t bitrack_hash(uint64_t lo, uint64_t hi) {
  uint64_t h = lo * 0x9E3779B97F4A7C15ULL + hi * 0xC2B2AE3D27D4EB4FULL;
  h ^= h >> 31;
  return h;
}

__device__ inline void bitrack_add(uint64_t& lo, uint64_t& hi, int letter) {
  if (letter < 16)
    lo += uint64_t(1) << (4 * letter);
  else
    hi += uint64_t(1) << (4 * (letter - 16));
}

// One open-addressing probe of a 3-word-slot table: the slot's meta word
// (start | count << 32), or 0 for a miss.
__device__ inline uint64_t probe_slots(const uint64_t* slots, uint64_t mask, uint64_t lo,
                                       uint64_t hi) {
  uint64_t i = bitrack_hash(lo, hi) & mask;
  for (;;) {
    const uint64_t* s = slots + i * 3;
    const uint64_t meta = s[2];
    if ((meta >> 32) == 0) return 0;
    if (s[0] == lo && s[1] == hi) return meta;
    i = (i + 1) & mask;
  }
}

// WordMap::lookup: the words of `len` with letter multiset (lo, hi), as the
// index of the first and their count.
__device__ inline uint32_t wm_lookup(int len, uint64_t lo, uint64_t hi, int* count) {
  const uint64_t meta = probe_slots(c_tables.slots + size_t(c_tables.slot_offset[len]) * 3,
                                    c_tables.slot_mask[len], lo, hi);
  *count = int(meta >> 32);
  return c_tables.word_base[len] + uint32_t(meta);
}

// The blank table for `nb` blanks: the entries of the words of `len` whose
// letters are the multiset (lo, hi) plus `nb` more.
__device__ inline const uint4* blank_lookup(int nb, int len, uint64_t lo, uint64_t hi, int* count) {
  const int b = nb - 1;
  const uint64_t meta =
    probe_slots(c_tables.blank_slots[b] + size_t(c_tables.blank_slot_offset[b][len]) * 3,
                c_tables.blank_slot_mask[b][len], lo, hi);
  *count = int(meta >> 32);
  return c_tables.blank_lists[b] + uint32_t(meta);
}

__device__ inline Word16 load_word(uint32_t index) {
  Word16 w;
  w.v = c_tables.words16[index];
  return w;
}

__device__ inline float leave_lookup(uint32_t key) {
  uint32_t h = (key * 0x9E3779B1u) >> c_tables.leave_shift;
  for (;;) {
    const uint32_t k = c_tables.leave_keys[h];
    if (k == key) return c_tables.leave_values[h];
    if (k == 0xFFFFFFFFu) return 0.0f;
    h = (h + 1) & c_tables.leave_mask;
  }
}

__device__ inline double peg_for_tiles(int tiles_played, int bag_size) {
  if (bag_size <= 0) return 0.0;
  return c_tables.peg[tiles_played * kPegBags + bag_size];
}

// ---- Candidates ----

// hasty_move_better: equity, then Macondo's tiebreaks.
__device__ bool better(const Cand& a, const Cand& b) {
  if (!b.valid) return a.valid != 0;
  if (!a.valid) return false;
  if (a.eq != b.eq) return a.eq > b.eq;
  if (a.score != b.score) return a.score > b.score;
  if (a.type != b.type) return a.type < b.type;
  if (a.row != b.row) return a.row < b.row;
  if (a.col != b.col) return a.col < b.col;
  if (a.vertical != b.vertical) return !a.vertical;
  if (a.n != b.n) return a.n < b.n;
  if (a.len != b.len) return a.len < b.len;
  if (a.t_hi != b.t_hi) return a.t_hi < b.t_hi;
  return a.t_lo < b.t_lo;
}

__device__ inline void pack_tiles(const uint8_t* bytes, int n, uint64_t& hi, uint64_t& lo) {
  hi = 0;
  lo = 0;
  for (int i = 0; i < n; ++i) {
    if (i < 8)
      hi |= uint64_t(bytes[i]) << (56 - 8 * i);
    else
      lo |= uint64_t(bytes[i]) << (56 - 8 * (i - 8));
  }
}

// The equity terms beyond the score for a play or exchange that keeps the
// leave of subrack mask `m`.
__device__ inline double equity_terms(const Smem& s, int m, int bag, int tiles_played,
                                      double opening) {
  const double lv = bag > 0 ? double(s.sub_lv[m]) : 0.0;
  double eg = 0.0;
  if (bag == 0) {
    const bool leave_empty = __popc(m) == s.rack_size;
    eg = leave_empty ? 2.0 * double(s.other_pv) : -2.0 * double(s.sub_pv[m]) - 10.0;
  }
  return lv + opening + peg_for_tiles(tiles_played, bag) + eg;
}

// build_play + HastyEquity::equity for `word` on `sp`, with the placed tiles
// whose word index is in `blank_bits` played as blanks. `m` is the subrack
// mask. Updates `best`.
__device__ void emit_play(const Smem& s, const Span& sp, const uint8_t* word, int len,
                          uint32_t blank_bits, int m, int bag, Cand& best) {
  const int v = span_view(sp), row = span_row(sp), wl = span_wl(sp);
  int main_letter_sum = 0, word_mult = 1, cross_total = 0, n = 0;
  uint16_t mask = 0;
  double opening = 0.0;
  uint8_t tiles[kMaxWordLen];
  uint8_t glyphs[kRackSize];
  for (int i = 0; i < len; ++i) {
    const int c = wl + i;
    const uint8_t g = sq_view(s, v, row, c);
    if (g != 0) {
      main_letter_sum += tile_value_of_glyph(g);
      tiles[i] = 0;
      continue;
    }
    const int L = word[i];
    const bool blank = (blank_bits >> i) & 1u;
    const int ni = nat_idx(v, row, c);
    const int lm = c_letter_mult[ni], wm = c_word_mult[ni];
    const int placed_v = (blank ? 0 : c_tile_values[L]) * lm;
    main_letter_sum += placed_v;
    word_mult *= wm;
    const int vi = row * kBoardSize + c;
    if (s.xnb[v][vi]) cross_total += (s.xscore[v][vi] + placed_v) * wm;
    glyphs[n++] = uint8_t(blank ? L + 27 : L + 1);
    mask |= uint16_t(1u << c);
    tiles[i] = uint8_t((L + 1) | (blank ? 0x80 : 0));
    if (s.num_tiles == 0 && (c == 2 || c == 6 || c == 8 || c == 12) && is_vowel(L)) opening -= 0.7;
  }
  int score = main_letter_sum * word_mult + cross_total;
  if (n == kRackSize) score += 50;
  const double eq = double(score) + equity_terms(s, m, bag, n, opening);
  if (best.valid && eq < best.eq) return;  // only a tie needs the full key
  Cand cd;
  cd.eq = eq;
  cd.score = int16_t(score);
  cd.valid = 1;
  cd.type = 0;
  cd.row = uint8_t(v ? wl : row);
  cd.col = uint8_t(v ? row : wl);
  cd.vertical = uint8_t(v);
  cd.n = uint8_t(n);
  cd.len = uint8_t(len);
  pack_tiles(tiles, len, cd.t_hi, cd.t_lo);
  cd.view = uint8_t(v);
  cd.vrow = uint8_t(row);
  cd.wl = uint8_t(wl);
  cd.mask = mask;
  for (int i = 0; i < kRackSize; ++i) cd.glyphs[i] = i < n ? glyphs[i] : 0;
  cd.pad = 0;
  if (better(cd, best)) best = cd;
}

// wmp_try_word's checks: filled squares match the word and placed letters
// pass their cross-checks, then the single-tile dedup rule.
__device__ bool word_fits(const Smem& s, const Span& sp, const uint8_t* word, int len) {
  const int v = span_view(sp), row = span_row(sp), wl = span_wl(sp);
  for (int i = 0; i < len; ++i) {
    const int c = wl + i;
    const uint8_t g = sq_view(s, v, row, c);
    if (g != 0) {
      if (glyph_letter(g) != word[i]) return false;
    } else if ((s.xmask[v][row * kBoardSize + c] & (1u << word[i])) == 0) {
      return false;
    }
  }
  if (v == 1 && span_placed(sp) == 1) {
    const int c = __ffs(sp.placed_mask) - 1;
    return s.xnb[1][row * kBoardSize + c] == 0;
  }
  return true;
}

// Every play of one blank-free item: subrack `m` of sorted_rack on `sp`. One
// word-map probe; each word found is tried at the span.
__device__ void process_item(const Smem& s, const Span& sp, int m, int bag, Cand& best,
                             WorkCounts& wc) {
  const int v = span_view(sp), row = span_row(sp), wl = span_wl(sp);
  const int len = span_wr(sp) - wl + 1;
  // The necessary condition that each placed square admits some letter of
  // the subrack.
  const uint32_t letters = s.sub_letters[m];
  for (uint16_t pm = sp.placed_mask; pm; pm &= uint16_t(pm - 1)) {
    const int c = __ffs(pm) - 1;
    if ((s.xmask[v][row * kBoardSize + c] & letters) == 0) return;
  }
  ++wc.items;
  int count;
  const uint32_t first = wm_lookup(len, sp.pt_lo + s.sub_lo[m], sp.pt_hi + s.sub_hi[m], &count);
  if (count == 0) return;
  ++wc.hits;
  wc.words += unsigned(count);
  for (int wi = 0; wi < count; ++wi) {
    const Word16 w = load_word(first + uint32_t(wi));
    if (!word_fits(s, sp, w.b, len)) continue;
    ++wc.plays;
    emit_play(s, sp, w.b, len, 0, m, bag, best);
  }
}

// One word of a blank item: the packed entry `e` (the word and the blanks'
// letters X and Y) tried at `sp` with the blanks at every placement of their
// letters.
__device__ void try_blank_word(const Smem& s, const Span& sp, int m, int nb, const uint4& e,
                               int bag, Cand& best, WorkCounts& wc) {
  const int wl = span_wl(sp);
  const int len = span_wr(sp) - wl + 1;
  const uint64_t lo = uint64_t(e.x) | (uint64_t(e.y) << 32);
  const uint64_t hi = uint64_t(e.z) | (uint64_t(e.w) << 32);
  const int X = int((hi >> (kBlankEntryXBit - 64)) & 31u);
  const int Y = int((hi >> (kBlankEntryYBit - 64)) & 31u);
  // A blank's letter must be admitted by some placed square.
  if (((sp.admitted >> X) & 1u) == 0) return;
  if (nb == 2 && ((sp.admitted >> Y) & 1u) == 0) return;
  ++wc.words;
  Word16 w;
  for (int i = 0; i < len; ++i) {
    const int bit = kBlankEntryLetterBits * i;
    uint64_t v;
    if (bit + kBlankEntryLetterBits <= 64)
      v = lo >> bit;
    else if (bit >= 64)
      v = hi >> (bit - 64);
    else
      v = (lo >> bit) | (hi << (64 - bit));
    w.b[i] = uint8_t(v & 31u);
  }
  if (!word_fits(s, sp, w.b, len)) return;
  for (int i = 0; i < len; ++i) {
    if (w.b[i] != X || ((sp.placed_mask >> (wl + i)) & 1u) == 0) continue;
    if (nb == 1) {
      ++wc.plays;
      emit_play(s, sp, w.b, len, 1u << i, m, bag, best);
      continue;
    }
    for (int j = (X == Y ? i + 1 : 0); j < len; ++j) {
      if (j == i || w.b[j] != Y || ((sp.placed_mask >> (wl + j)) & 1u) == 0) continue;
      ++wc.plays;
      emit_play(s, sp, w.b, len, (1u << i) | (1u << j), m, bag, best);
    }
  }
}

// Inserts `x` into the `n` values of `a`, kept highest first.
__device__ inline void insert_descending(int* a, int& n, int x) {
  int j = n++;
  for (; j > 0 && a[j - 1] < x; --j) a[j] = a[j - 1];
  a[j] = x;
}

// ---- Ply phases ----

// The face value of player `p`'s rack (blanks count 0).
__device__ inline int rack_value(const Smem& s, int p) {
  int pv = 0;
  for (int t = 0; t < 26; ++t) pv += s.rack[p][t] * c_tile_values[t];
  return pv;
}

// The mover's rack sorted ascending by tile (blanks last), and the subrack
// masks grouped by size and blank count. One thread.
__device__ void prepare_rack(Smem& s) {
  int n = 0;
  for (int t = 0; t < kTileKinds; ++t) {
    for (int c = 0; c < s.rack[s.mover][t]; ++c) s.sorted_rack[n++] = uint8_t(t);
  }
  s.rack_size = n;
  s.rack_blanks = s.rack[s.mover][kBlankTile];
  for (int k = 0; k <= kRackSize; ++k)
    for (int b = 0; b <= kMaxBlanks; ++b) s.n_masks[k][b] = 0;
  const int blank_bits = ((1 << s.rack_blanks) - 1) << (n - s.rack_blanks);
  for (int m = 1; m < (1 << n); ++m) {
    const int k = __popc(m), b = __popc(m & blank_bits);
    s.masks[k][b][s.n_masks[k][b]++] = uint8_t(m);
  }
  s.other_pv = rack_value(s, 1 - s.mover);
  uint32_t letters = s.rack_blanks > 0 ? kAllLetters : 0u;
  for (int t = 0; t < 26; ++t) {
    if (s.rack[s.mover][t] > 0) letters |= 1u << t;
  }
  s.rack_letters = letters;
  s.nspans = 0;
  s.next_item = 0;
  s.next_itemb = 0;
}

// Thread m: subrack m's letters; the leave after taking it, its value and
// point value; and, with seven or more tiles in the bag, the exchange of
// subrack m as a candidate (generate_legal_exchanges).
__device__ void price_subracks(Smem& s, int m, int bag, Cand& best) {
  if (m >= (1 << s.rack_size)) return;
  uint32_t key = 0, letters = 0;
  uint64_t lo = 0, hi = 0;
  int n_leave = 0, pv = 0;
  for (int i = 0; i < s.rack_size; ++i) {
    const int t = s.sorted_rack[i];
    if ((m >> i) & 1) {
      if (t != kBlankTile) {
        bitrack_add(lo, hi, t);
        letters |= 1u << t;
      }
      continue;
    }
    key |= uint32_t(t) << (5 * n_leave++);
    if (t != kBlankTile) pv += c_tile_values[t];
  }
  for (int i = n_leave; i < 6; ++i) key |= 31u << (5 * i);
  s.sub_lv[m] = n_leave > 0 ? leave_lookup(key) : 0.0f;
  int vals[kRackSize], nv = 0;
  for (int i = 0; i < s.rack_size; ++i) {
    if (((m >> i) & 1) == 0) continue;
    const int t = s.sorted_rack[i];
    insert_descending(vals, nv, t == kBlankTile ? 0 : c_tile_values[t]);
  }
  uint32_t packed_vals = 0;
  for (int i = 0; i < nv; ++i) packed_vals |= uint32_t(vals[i]) << (4 * i);
  s.sub_values[m] = packed_vals;
  s.sub_pv[m] = int16_t(pv);
  s.sub_lo[m] = lo;
  s.sub_hi[m] = hi;
  s.sub_letters[m] = letters;
  if (m == 0 || bag < kRackSize) return;
  const int k = __popc(m);
  Cand cd;
  cd.eq = equity_terms(s, m, bag, k, 0.0);
  cd.score = 0;
  cd.valid = 1;
  cd.type = 1;
  cd.row = cd.col = cd.vertical = 0;
  cd.n = uint8_t(k);
  cd.len = uint8_t(k);
  // Macondo orders an exchange's tiles with the blank as 0, so blanks first.
  uint8_t tiles[kRackSize];
  int nt = 0;
  for (int i = 0; i < s.rack_size; ++i) {
    if (((m >> i) & 1) && s.sorted_rack[i] == kBlankTile) tiles[nt++] = 0;
  }
  for (int i = 0; i < s.rack_size; ++i) {
    if (((m >> i) & 1) && s.sorted_rack[i] != kBlankTile)
      tiles[nt++] = uint8_t(s.sorted_rack[i] + 1);
  }
  pack_tiles(tiles, nt, cd.t_hi, cd.t_lo);
  cd.view = cd.vrow = cd.wl = 0;
  cd.mask = 0;
  int ng = 0;
  for (int i = 0; i < s.rack_size; ++i) {
    if ((m >> i) & 1) cd.glyphs[ng++] = s.sorted_rack[i] == kBlankTile ? 53 : s.sorted_rack[i] + 1;
  }
  for (int i = ng; i < kRackSize; ++i) cd.glyphs[i] = 0;
  cd.pad = 0;
  if (better(cd, best)) best = cd;
}

// wmp_emit_leftmost_anchor's spans of every lane into the scratch: a warp
// per lane, a lane per leftmost anchor A.
__device__ void enumerate_spans(Smem& s, Scratch& sc, int warp, int lane) {
  const bool empty_board = s.num_tiles == 0;
  for (int lr = warp; lr < kLanes; lr += kWarps) {
    const int view = lr / kBoardSize, row = lr % kBoardSize;
    if (lane >= kBoardSize) continue;
    const int A = lane;
    if (sq_view(s, view, row, A) != 0) continue;
    int wl = A;
    uint64_t pt_lo = 0, pt_hi = 0;
    while (wl - 1 >= 0 && sq_view(s, view, row, wl - 1) != 0) {
      --wl;
      bitrack_add(pt_lo, pt_hi, glyph_letter(sq_view(s, view, row, wl)));
    }
    int placed = 0;
    bool any_playthrough = wl < A, any_cross = false;
    uint32_t admitted = 0;
    uint16_t placed_mask = 0;
    for (int wr = A; wr < kBoardSize; ++wr) {
      const uint8_t g = sq_view(s, view, row, wr);
      if (g != 0) {
        any_playthrough = true;
        bitrack_add(pt_lo, pt_hi, glyph_letter(g));
      } else {
        const int vi = row * kBoardSize + wr;
        // No rack letter fits here, so no longer span has a play either
        // (wmp_placeable_squares).
        if ((s.xmask[view][vi] & s.rack_letters) == 0) break;
        ++placed;
        placed_mask |= uint16_t(1u << wr);
        if (s.xnb[view][vi]) any_cross = true;
        admitted |= s.xmask[view][vi];
      }
      if (placed > s.rack_size) break;
      if (wr + 1 < kBoardSize && sq_view(s, view, row, wr + 1) != 0) continue;
      if (wr - wl + 1 < 2) continue;
      const bool connected =
        empty_board ? (row == 7 && A <= 7 && wr >= 7) : (any_playthrough || any_cross);
      if (!connected) continue;
      const int idx = atomicAdd(&s.nspans, 1);
      Span sp;
      sp.pt_lo = pt_lo;
      sp.pt_hi = pt_hi;
      sp.admitted = admitted & kAllLetters;
      sp.placed_mask = placed_mask;
      sp.packed = uint16_t(view | (row << 1) | (wl << 5) | (wr << 9) | (placed << 13));
      sc.spans[idx] = sp;
    }
  }
}

// A block exclusive scan of per-thread values: warp shuffles, then the warp
// totals through `warp_sum`. Returns the thread's exclusive prefix.
__device__ int block_exclusive_scan(int value, int tid, int* warp_sum) {
  const int lane = tid & 31, warp = tid >> 5;
  int incl = value;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, offset);
    if (lane >= offset) incl += v;
  }
  if (lane == 31) warp_sum[warp] = incl;
  __syncthreads();
  int warp_base = 0;
  for (int w = 0; w < warp; ++w) warp_base += warp_sum[w];
  return warp_base + incl - value;
}

// off / offb: the exclusive prefix sums of the two item counts over the
// spans, a contiguous chunk of spans per thread. Sets nitems / nitemsb.
__device__ void scan_items(Smem& s, Scratch& sc, int tid) {
  const int n = s.nspans;
  const int chunk = (n + kThreads - 1) / kThreads;
  const int lo = tid * chunk, hi = min(lo + chunk, n);
  int sum = 0, sumb = 0;
  for (int i = lo; i < hi; ++i) {
    const int k = span_placed(sc.spans[i]);
    sum += s.n_masks[k][0];
    sumb += s.n_masks[k][1] + s.n_masks[k][2];
  }
  int run = block_exclusive_scan(sum, tid, s.warp_sum);
  int runb = block_exclusive_scan(sumb, tid, s.warp_sumb);
  for (int i = lo; i < hi; ++i) {
    const int k = span_placed(sc.spans[i]);
    sc.off[i] = uint32_t(run);
    sc.offb[i] = uint32_t(runb);
    run += s.n_masks[k][0];
    runb += s.n_masks[k][1] + s.n_masks[k][2];
  }
  if (tid == kThreads - 1) {
    s.nitems = run;
    s.nitemsb = runb;
    sc.off[n] = uint32_t(run);
    sc.offb[n] = uint32_t(runb);
  }
}

// The span holding `item` under the exclusive offsets `off`.
__device__ int find_span(int nspans, const uint32_t* off, uint32_t item) {
  int lo = 0, hi = nspans - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (off[mid] <= item)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// ---- The span bound ----
// An upper bound on the equity of any play on a span, so the second pass can
// skip spans that cannot beat a play or exchange already found. Every play on
// the span places its `placed` squares from a subrack of that many tiles:
// its score is the playthrough tiles' sum times the word multiplier, the
// cross words' fixed scores, the bingo, and each placed tile's value times
// its square's coefficient (letter multiplier times the word multiplier,
// plus the cross word's multiplier where it has one), which the rack's
// highest values on the highest coefficients bound (the rearrangement
// inequality). Its leave is at best leave_bound, and the pre-endgame term is
// fixed by the tiles played; the opening penalty only lowers it. Bounds are
// used only with tiles in the bag (no endgame terms).

// A margin under which a bound counts as tying the bar: a play whose equity
// equals the bar could still win on the tiebreaks, and the bound and the
// equity sum their terms in different orders.
constexpr double kBoundSlack = 1e-6;
// The first pass takes the spans whose bound is within this of the best
// bound, to find a strong play to prune the rest against.
constexpr double kFirstPassMargin = 20.0;

// leave_bound[k]: the best leave value among subracks of k tiles. Threads
// 0..rack_size.
__device__ void bound_leaves(Smem& s, int tid) {
  if (tid > s.rack_size) return;
  float best = tid == s.rack_size ? 0.0f : -1e30f;
  for (int b = 0; b <= kMaxBlanks; ++b)
    for (int j = 0; j < s.n_masks[tid][b]; ++j) best = fmaxf(best, s.sub_lv[s.masks[tid][b][j]]);
  s.leave_bound[tid] = best;
}

__device__ SpanParts span_parts(const Smem& s, const Span& sp) {
  const int v = span_view(sp), row = span_row(sp), wl = span_wl(sp), wr = span_wr(sp);
  int playthrough = 0, word_mult = 1;
  for (int c = wl; c <= wr; ++c) {
    const uint8_t g = sq_view(s, v, row, c);
    if (g != 0)
      playthrough += tile_value_of_glyph(g);
    else
      word_mult *= c_word_mult[nat_idx(v, row, c)];
  }
  int coef[kRackSize];
  int n = 0, fixed = 0;
  for (int c = wl; c <= wr; ++c) {
    if (sq_view(s, v, row, c) != 0) continue;
    const int ni = nat_idx(v, row, c), vi = row * kBoardSize + c;
    const int lm = c_letter_mult[ni], wm = c_word_mult[ni];
    const bool cross = s.xnb[v][vi] != 0;
    if (cross) fixed += s.xscore[v][vi] * wm;
    insert_descending(coef, n, lm * (word_mult + (cross ? wm : 0)));
  }
  SpanParts p{playthrough * word_mult + fixed + (n == kRackSize ? 50 : 0), 0};
  for (int i = 0; i < n; ++i) p.coef |= uint64_t(coef[i]) << (8 * i);
  return p;
}

// The score bound of the `k` highest face values of `values` (4 bits each,
// highest first) on a span with parts `p`.
__device__ inline int score_bound(const SpanParts& p, uint32_t values, int k) {
  int score = p.fixed;
  for (int i = 0; i < k; ++i)
    score += int((p.coef >> (8 * i)) & 0xFF) * int((values >> (4 * i)) & 0xF);
  return score;
}

// The bound of any play of subrack `m` on a span: its score bound, its leave,
// and the pre-endgame term.
__device__ inline double item_bound(const Smem& s, const SpanParts& p, int m, int k, int bag) {
  return double(score_bound(p, s.sub_values[m], k)) + double(s.sub_lv[m]) + peg_for_tiles(k, bag);
}

// The bound of any play on a span of `k` placed squares: the rack's highest
// values on its highest coefficients, and the best leave of `k` tiles played.
__device__ inline double span_bound(const Smem& s, const SpanParts& p, int k, int bag) {
  const uint32_t rack_values = s.sub_values[(1 << s.rack_size) - 1];
  return double(score_bound(p, rack_values, k)) + double(s.leave_bound[k]) + peg_for_tiles(k, bag);
}

// The block's maximum of `value` into `out` (all threads take part).
__device__ void block_max(double value, int tid, double* warp_max, double* out) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
  if ((tid & 31) == 0) warp_max[tid >> 5] = value;
  __syncthreads();
  if (tid == 0) {
    double m = warp_max[0];
    for (int w = 1; w < kWarps; ++w) m = fmax(m, warp_max[w]);
    *out = m;
  }
  __syncthreads();
}

// The block's best equity found so far into `out`.
__device__ inline void best_equity(Smem& s, int tid, const Cand& local, double* out) {
  block_max(local.valid ? local.eq : -1e300, tid, s.warp_max, out);
}

// Bounds every span and partitions them into Scratch::parted: the first
// pass's (bound within kFirstPassMargin of the best) at the front, which
// then replace the span list; the rest, counted in s.nrest, after them.
__device__ void bound_spans(Smem& s, Scratch& sc, int tid, int bag) {
  const int n = s.nspans;
  double top = -1e300;
  for (int i = tid; i < n; i += kThreads) {
    sc.parts[i] = span_parts(s, sc.spans[i]);
    sc.bound[i] = span_bound(s, sc.parts[i], span_placed(sc.spans[i]), bag);
    top = fmax(top, sc.bound[i]);
  }
  block_max(top, tid, s.warp_max, &s.best_bound);
  if (tid == 0) s.nfront = s.nrest = 0;
  __syncthreads();
  const double cut = s.best_bound - kFirstPassMargin;
  for (int i = tid; i < n; i += kThreads) {
    const bool first = sc.bound[i] >= cut;
    const int j = first ? atomicAdd(&s.nfront, 1) : n - 1 - atomicAdd(&s.nrest, 1);
    sc.parted[j] = BoundedSpan{sc.spans[i], sc.parts[i], sc.bound[i]};
  }
  __syncthreads();
  for (int i = tid; i < s.nfront; i += kThreads) {
    sc.spans[i] = sc.parted[i].span;
    sc.parts[i] = sc.parted[i].parts;
  }
  __syncthreads();  // every thread has read the count
  if (tid == 0) s.nspans = s.nfront;
  __syncthreads();
}

// The rest of the spans whose bound reaches `bar` (the best equity found so
// far) become the span list; the others cannot hold the ply's best move.
__device__ void keep_reachable(Smem& s, Scratch& sc, int tid, double bar) {
  const int n = s.nfront + s.nrest;
  __syncthreads();  // every thread has read the counts
  if (tid == 0) s.nspans = 0;
  __syncthreads();
  for (int i = s.nfront + tid; i < n; i += kThreads) {
    const BoundedSpan& b = sc.parted[i];
    if (b.bound + kBoundSlack < bar) continue;
    const int j = atomicAdd(&s.nspans, 1);
    sc.spans[j] = b.span;
    sc.parts[j] = b.parts;
  }
  if (tid == 0) {
    s.next_item = 0;
    s.next_itemb = 0;
  }
  __syncthreads();
}

// With tiles in the bag: the span bounds, the first pass's spans as the span
// list, and the best exchange as its bar.
__device__ void start_bounded_ply(Smem& s, Scratch& sc, int tid, int bag, const Cand& local) {
  bound_leaves(s, tid);
  best_equity(s, tid, local, &s.bar);
  bound_spans(s, sc, tid, bag);
}

// After the first pass: the best equity it found becomes the bar, and the
// rest of the spans that can still reach it the span list.
__device__ void start_second_pass(Smem& s, Scratch& sc, int tid, const Cand& local) {
  best_equity(s, tid, local, &s.bar);
  keep_reachable(s, sc, tid, s.bar);
}

// The ply's blank-free items, a thread per item off the shared queue.
__device__ void process_items(Smem& s, const Scratch& sc, int bag, Cand& best, WorkCounts& wc) {
  for (;;) {
    const uint32_t item = uint32_t(atomicAdd(&s.next_item, 1));
    if (item >= uint32_t(s.nitems)) break;
    const int si = find_span(s.nspans, sc.off, item);
    const Span sp = sc.spans[si];
    const int k = span_placed(sp), m = s.masks[k][0][item - sc.off[si]];
    if (item_bound(s, sc.parts[si], m, k, bag) + kBoundSlack < s.bar) continue;
    process_item(s, sp, m, bag, best, wc);
  }
}

// The ply's blank items, 32 per warp off the shared queue: each lane probes
// the blank table for its own item, then the lanes share the words of all 32
// hits as one stream, so a warp keeps 32 loads in flight at every step
// instead of walking one hit list at a time.
__device__ void process_blank_items(Smem& s, const Scratch& sc, int tid, int bag, Cand& best,
                                    WorkCounts& wc) {
  const int lane = tid & 31, warp = tid >> 5;
  Smem::WarpItems& wi = s.warp_items[warp];
  const unsigned full = 0xffffffffu;
  for (;;) {
    uint32_t base = 0;
    if (lane == 0) base = uint32_t(atomicAdd(&s.next_itemb, 32));
    base = __shfl_sync(full, base, 0);
    if (base >= uint32_t(s.nitemsb)) break;
    const uint32_t item = base + uint32_t(lane);
    int count = 0;
    if (item < uint32_t(s.nitemsb)) {
      const int si = find_span(s.nspans, sc.offb, item);
      const Span sp = sc.spans[si];
      const int k = span_placed(sp);
      uint32_t j = item - sc.offb[si];
      int nb = 1;
      if (j >= s.n_masks[k][1]) {
        j -= s.n_masks[k][1];
        nb = 2;
      }
      const int m = s.masks[k][nb][j];
      const int len = span_wr(sp) - span_wl(sp) + 1;
      if (item_bound(s, sc.parts[si], m, k, bag) + kBoundSlack >= s.bar) {
        wi.list[lane] =
          blank_lookup(nb, len, sp.pt_lo + s.sub_lo[m], sp.pt_hi + s.sub_hi[m], &count);
        ++wc.items;
        wc.hits += count > 0 ? 1u : 0u;
      }
      wi.span[lane] = sp;
      wi.m[lane] = uint8_t(m);
      wi.nb[lane] = uint8_t(nb);
    }
    // Exclusive scan of the counts across the lanes.
    int incl = count;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int v = __shfl_up_sync(full, incl, offset);
      if (lane >= offset) incl += v;
    }
    const int excl = incl - count;
    const int total = __shfl_sync(full, incl, 31);
    __syncwarp();
    for (int t0 = 0; t0 < total; t0 += 32) {
      const int t = t0 + lane;
      // The lane whose hit list holds word t: the last with excl <= t.
      int lo = 0, hi = 31;
      for (int step = 0; step < 5; ++step) {
        const int mid = (lo + hi + 1) >> 1;
        const int v = __shfl_sync(full, excl, mid);
        if (v <= t)
          lo = mid;
        else
          hi = mid - 1;
      }
      const int off = __shfl_sync(full, excl, lo);
      if (t >= total) continue;
      const uint4 e = wi.list[lo][t - off];
      try_blank_word(s, wi.span[lo], wi.m[lo], wi.nb[lo], e, bag, best, wc);
    }
    __syncwarp();
  }
}

__device__ inline Cand shfl_down_cand(const Cand& c, int offset) {
  CandWords in, out;
  in.c = c;
  for (int i = 0; i < 12; ++i) out.w[i] = __shfl_down_sync(0xffffffffu, in.w[i], offset);
  return out.c;
}

// The block's best candidate into s.chosen: a shuffle reduction per warp,
// then thread 0 over the warps.
__device__ void reduce_best(Smem& s, int tid, Cand local) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const Cand other = shfl_down_cand(local, offset);
    if (better(other, local)) local = other;
  }
  if ((tid & 31) == 0) s.warp_best[tid >> 5] = local;
  __syncthreads();
  if (tid == 0) {
    Cand b = s.warp_best[0];
    for (int w = 1; w < kWarps; ++w) {
      if (better(s.warp_best[w], b)) b = s.warp_best[w];
    }
    s.chosen = b;
  }
  __syncthreads();
}

__device__ inline int rack_total(const Smem& s, int p) {
  int n = 0;
  for (int t = 0; t < kTileKinds; ++t) n += s.rack[p][t];
  return n;
}

// Records `player`'s rack as its leave: called after its move takes its
// tiles, before it draws.
__device__ inline void record_leave(Smem& s, int player) {
  for (int t = 0; t < kTileKinds; ++t) s.leave[player][t] = s.rack[player][t];
}

// With the bag empty, a second scoreless turn in a row ends the game as
// rollout_abi.h describes: each player loses its rack's value.
__device__ void end_if_stalemate(Smem& s) {
  if (s.bag.remaining() > 0 || s.scoreless < 2) return;
  const int v0 = rack_value(s, 0), v1 = rack_value(s, 1);
  s.scores[0] -= v0;
  s.scores[1] -= v1;
  s.game_over = 1;
}

// Applies s.chosen for the mover: tiles, score, draw, play-out bonus,
// stalemate. One thread; the cross-check update follows in parallel.
__device__ void apply_chosen(Smem& s, uint64_t seed, uint32_t rollout) {
  const Cand& c = s.chosen;
  const int mover = s.mover;
  if (!c.valid) {
    record_leave(s, mover);
    ++s.scoreless;
    end_if_stalemate(s);
    return;
  }
  if (c.type == 0) {
    int gi = 0;
    for (int col = 0; col < kBoardSize; ++col) {
      if (((c.mask >> col) & 1u) == 0) continue;
      const uint8_t g = c.glyphs[gi];
      const int r = c.view ? col : c.vrow, cc = c.view ? c.vrow : col;
      s.sq[r * kBoardSize + cc] = g;
      s.placed_r[gi] = uint8_t(r);
      s.placed_c[gi] = uint8_t(cc);
      --s.rack[mover][glyph_is_blank(g) ? kBlankTile : glyph_letter(g)];
      ++gi;
    }
    s.num_tiles += c.n;
    s.scores[mover] += c.score;
    s.scoreless = 0;
    record_leave(s, mover);
    const int draws = min(int(c.n), s.bag.remaining());
    for (int i = 0; i < draws; ++i) ++s.rack[mover][s.bag.draw(seed, rollout)];
    if (rack_total(s, mover) == 0) {
      s.scores[mover] += 2 * s.other_pv;
      s.game_over = 1;
    }
  } else {
    for (int i = 0; i < c.n; ++i) {
      --s.rack[mover][c.glyphs[i] == 53 ? kBlankTile : c.glyphs[i] - 1];
    }
    record_leave(s, mover);
    for (int i = 0; i < c.n; ++i) ++s.rack[mover][s.bag.draw(seed, rollout)];
    for (int i = 0; i < c.n; ++i)
      s.bag.put_back(uint8_t(c.glyphs[i] == 53 ? kBlankTile : c.glyphs[i] - 1));
    ++s.scoreless;
  }
}

// Board::update_caches_after_place: the placed squares' own entries, then the
// empty squares past each end of the perpendicular run through each placed
// tile, in both views.
__device__ void update_cross_checks(Smem& s, int tid) {
  const int n = s.chosen.n;
  if (tid < 2 * n) {
    const int view = tid / n, i = tid % n;
    const int vr = view ? s.placed_c[i] : s.placed_r[i];
    const int vc = view ? s.placed_r[i] : s.placed_c[i];
    s.xmask[view][vr * kBoardSize + vc] = kAllLetters;
    s.xscore[view][vr * kBoardSize + vc] = 0;
    s.xnb[view][vr * kBoardSize + vc] = 0;
  }
  __syncthreads();
  const int warp = tid >> 5, lane = tid & 31;
  for (int task = warp; task < 4 * n; task += kWarps) {
    const int view = task / (2 * n), rest = task % (2 * n), i = rest / 2, end = rest % 2;
    const int vr = view ? s.placed_c[i] : s.placed_r[i];
    const int vc = view ? s.placed_r[i] : s.placed_c[i];
    int top = vr;
    while (top - 1 >= 0 && sq_view(s, view, top - 1, vc) != 0) --top;
    int bot = vr;
    while (bot + 1 < kBoardSize && sq_view(s, view, bot + 1, vc) != 0) ++bot;
    const int r = end == 0 ? top - 1 : bot + 1;
    if (r >= 0 && r < kBoardSize) recompute_cross_check(s, view, r, vc, lane);
  }
  __syncthreads();
}

__device__ void write_move(const Cand& c, PackedMove& out) {
  if (!c.valid) {
    out = PackedMove{};
    return;
  }
  out.type = c.type;
  out.horizontal = c.type == 0 ? (c.view == 0 ? 1 : 0) : 1;
  out.start = c.type == 0 ? c.vrow : 0;
  out.num_glyphs = c.n;
  for (int i = 0; i < kRackSize; ++i) out.glyphs[i] = c.glyphs[i];
  out.pad = 0;
  out.square_mask = c.type == 0 ? c.mask : 0;
  out.score = uint16_t(c.score);
}

// Thread 0's phase clock: adds the cycles since `t0` to phase `p`.
struct PhaseClock {
  long long t0;
  unsigned long long acc[kPhases];
  __device__ void start() { t0 = clock64(); }
  __device__ void lap(int p) {
    const long long t = clock64();
    acc[p] += uint64_t(t - t0);
    t0 = t;
  }
};

// One pass over the span list's items: the item scan, then the probes of the
// items whose bound reaches s.bar.
__device__ void run_items_pass(Smem& s, Scratch& sc, int tid, int bag, Cand& local, PhaseClock& pc,
                               WorkCounts& wc) {
  scan_items(s, sc, tid);
  __syncthreads();
  if (tid == 0) pc.lap(kPhaseSpans);
  process_items(s, sc, bag, local, wc);
  process_blank_items(s, sc, tid, bag, local, wc);
  __syncthreads();
  if (tid == 0) pc.lap(kPhaseItems);
}

// One ply by s.mover.
__device__ void play_ply(Smem& s, Scratch& sc, int tid, uint64_t seed, uint32_t rollout,
                         PackedMove& out, PhaseClock& pc, WorkCounts& wc) {
  const int warp = tid / 32, lane = tid % 32;
  if (tid == 0) prepare_rack(s);
  __syncthreads();
  if (s.rack_size == 0) {
    if (tid == 0) {
      s.chosen.valid = 0;
      out = PackedMove{};
    }
    __syncthreads();
    return;
  }
  const int bag = s.bag.remaining();
  Cand local;
  local.valid = 0;
  if (tid < 128) price_subracks(s, tid, bag, local);
  enumerate_spans(s, sc, warp, lane);
  __syncthreads();
  if (tid == 0) {
    pc.lap(kPhaseLeaves);
    atomicAdd(&g_counts[kCountSpans], (unsigned long long)s.nspans);
  }
  // With tiles in the bag, two passes: the spans most likely to hold the
  // best move, then the rest that can still beat what the first found.
  const bool bounded = bag > 0;
  if (bounded) {
    start_bounded_ply(s, sc, tid, bag, local);
  } else if (tid == 0) {
    s.bar = -1e300;
  }
  run_items_pass(s, sc, tid, bag, local, pc, wc);
  if (bounded) {
    start_second_pass(s, sc, tid, local);
    run_items_pass(s, sc, tid, bag, local, pc, wc);
  }
  reduce_best(s, tid, local);
  if (tid == 0) {
    write_move(s.chosen, out);
    apply_chosen(s, seed, rollout);
    pc.lap(kPhaseApply);
  }
  __syncthreads();
  if (s.chosen.valid && s.chosen.type == 0) update_cross_checks(s, tid);
  if (tid == 0) pc.lap(kPhaseCross);
}

// A free scratch slot, by compare-and-swap over the busy flags from a start
// spread over blocks.
__device__ int acquire_scratch(uint32_t rollout) {
  for (uint32_t i = rollout;; ++i) {
    const int slot = int(i % kScratchSlots);
    if (atomicCAS(&c_tables.scratch_busy[slot], 0, 1) == 0) return slot;
  }
}

// Copies the start's board and cross-checks into shared memory, the whole
// block taking part.
__device__ void load_start(const PositionAbi& pos, Smem& s) {
  for (int i = threadIdx.x; i < kSquares; i += kThreads) {
    s.sq[i] = pos.squares[i];
    for (int v = 0; v < 2; ++v) {
      s.xmask[v][i] = pos.cross_mask[v][i];
      s.xscore[v][i] = pos.cross_score[v][i];
      s.xnb[v][i] = pos.cross_has_neighbor[v][i];
    }
  }
}

// The racks and the bag, in rollout_abi.h's order (hasty_rollout_task.cpp's
// start_rollout). Thread 0, after the scores are set.
__device__ void deal(const PositionAbi& pos, const GroupAbi& group, const OppLeaveAbi* leaves,
                     uint32_t rollout, Smem& s) {
  for (int t = 0; t < kTileKinds; ++t) {
    s.rack[0][t] = pos.leave[t];
    s.rack[1][t] = 0;
  }
  for (int i = 0; i < group.pool_size; ++i) s.bag.tiles[i] = group.pool[i];
  s.bag.len = group.pool_size;
  s.bag.cursor = 0;
  s.bag.next_rand = 0;
  if (rollout < group.num_leaves) {
    const OppLeaveAbi& leave = leaves[group.leave_base + rollout];
    for (int i = 0; i < leave.n; ++i) {
      ++s.rack[1][leave.tiles[i]];
      remove_from_bag(s.bag, leave.tiles[i]);
    }
  }
  for (int k = 0; k < 2; ++k) {
    const int p = 1 - k;  // the opponent first
    const int draws = refill_draw(s.bag.remaining(), rack_total(s, p));
    for (int i = 0; i < draws; ++i) ++s.rack[p][s.bag.draw(group.seed, rollout)];
  }
  for (int i = 0; i < pos.num_returned; ++i) s.bag.put_back(pos.returned[i]);
  // A player who never moves keeps its full rack as its leave.
  record_leave(s, 0);
  record_leave(s, 1);
  // Our move went out with the bag empty: the game ended with it.
  if (rack_total(s, 0) == 0) {
    s.scores[0] += 2 * rack_value(s, 1);
    s.game_over = 1;
  }
}

__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm)
  rollout_kernel(const RolloutTask* tasks, const PositionAbi* positions, const GroupAbi* groups,
                 const OppLeaveAbi* opp_leaves, RolloutResult* out, LeafState* leaves) {
  __shared__ Smem s;
  const int tid = threadIdx.x;
  const RolloutTask task = tasks[blockIdx.x];
  const PositionAbi& pos = positions[task.position];
  const GroupAbi& group = groups[pos.group];
  const uint32_t rollout = task.rollout;
  const uint64_t seed = group.seed;
  PhaseClock pc;
  WorkCounts wc;
  for (int p = 0; p < kPhases; ++p) pc.acc[p] = 0;
  if (tid == 0) pc.start();
  load_start(pos, s);
  if (tid == 0) {
    s.slot = acquire_scratch(blockIdx.x);
    s.scores[0] = pos.scores[0];
    s.scores[1] = pos.scores[1];
    s.num_tiles = pos.num_tiles;
    s.game_over = 0;
    s.scoreless = 0;
    s.overflow = 0;
    deal(pos, group, opp_leaves, rollout, s);
  }
  __syncthreads();
  Scratch& sc = c_tables.scratch[s.slot];
  if (tid == 0) pc.lap(kPhaseInit);
  RolloutResult res;
  PackedMove later;  // the plies past the first two, which the result doesn't report
  // Two plies, then to the game's end if the bag is empty (rollout_abi.h).
  for (int ply = 0; ply < kMaxRolloutPlies && !s.game_over && (ply < 2 || s.bag.remaining() == 0);
       ++ply) {
    if (tid == 0) s.mover = ply % 2 == 0 ? 1 : 0;
    __syncthreads();
    play_ply(s, sc, tid, seed, rollout, ply < 2 ? res.moves[ply] : later, pc, wc);
    __syncthreads();
    if (tid == 0) atomicAdd(&g_counts[kCountPlies], 1ull);
  }
  // The work counters: a warp sum, then one atomic per warp.
  unsigned sums[4] = {wc.items, wc.hits, wc.words, wc.plays};
  for (int i = 0; i < 4; ++i) {
    for (int offset = 16; offset > 0; offset >>= 1)
      sums[i] += __shfl_down_sync(0xffffffffu, sums[i], offset);
  }
  if ((tid & 31) == 0) {
    atomicAdd(&g_counts[kCountItems], (unsigned long long)sums[0]);
    atomicAdd(&g_counts[kCountHits], (unsigned long long)sums[1]);
    atomicAdd(&g_counts[kCountWords], (unsigned long long)sums[2]);
    atomicAdd(&g_counts[kCountPlays], (unsigned long long)sums[3]);
  }
  // The leaf: the end state, left on the device for a leaf evaluator.
  LeafState& leaf = leaves[blockIdx.x];
  for (int i = tid; i < kSquares; i += kThreads) {
    leaf.squares[i] = s.sq[i];
    leaf.cross_mask[0][i] = s.xmask[0][i];
    leaf.cross_mask[1][i] = s.xmask[1][i];
  }
  if (tid < 2 * kTileKinds) {
    leaf.racks[tid / kTileKinds][tid % kTileKinds] = s.rack[tid / kTileKinds][tid % kTileKinds];
    leaf.leaves[tid / kTileKinds][tid % kTileKinds] = s.leave[tid / kTileKinds][tid % kTileKinds];
  }
  if (tid == 0) {
    leaf.scores[0] = s.scores[0];
    leaf.scores[1] = s.scores[1];
    leaf.bag_size = uint8_t(s.bag.remaining());
    leaf.game_over = uint8_t(s.game_over);
    res.game_over = uint8_t(s.game_over);
    res.pad = uint8_t(s.overflow);
    res.diff = int16_t(s.scores[0] - s.scores[1]);
    out[blockIdx.x] = res;
    for (int p = 0; p < kPhases; ++p) atomicAdd(&g_phase_cycles[p], pc.acc[p]);
    __threadfence();
    atomicExch(&c_tables.scratch_busy[s.slot], 0);
  }
}

// ---- The leaf encoder ----
// LeafEvalInputEncoder on the device: one block per leaf, writing its row
// of kLeafSpatialPlanes x 225 floats and kLeafScalars floats, from player
// 0's side (the player who moved last). It follows
// encoding/leaf_eval_input_encoder.cpp and board_planes.cpp plane for plane.

constexpr int kEncodeThreads = 256;

// LeaveValues::lookup of the leave `counts` holds: 0 for an empty leave or
// one too long for the table.
__device__ float leave_value_of_counts(const uint8_t* counts) {
  uint32_t key = 0;
  int n = 0;
  for (int t = 0; t < kTileKinds; ++t) {
    for (int c = 0; c < counts[t]; ++c) {
      if (n < 6) key |= uint32_t(t) << (5 * n);
      ++n;
    }
  }
  if (n == 0 || n > 6) return 0.0f;
  for (int i = n; i < 6; ++i) key |= 31u << (5 * i);
  return leave_lookup(key);
}

// The join planes' seat fit (leaf_eval_input_encoder.cpp's seat_fit): how
// many of the seat's tiles satisfy the cross-check `mask`, and the highest
// face value among them. A blank fits any square that allows some letter.
struct SeatFit {
  int fitting;
  int max_value;
};

__device__ SeatFit seat_fit(const int* counts, uint32_t mask) {
  SeatFit fit{0, 0};
  for (int t = 0; t < kTileKinds; ++t) {
    const int n = counts[t];
    if (n == 0) continue;
    const bool fits = t == kBlankTile ? mask != 0 : ((mask >> t) & 1u) != 0;
    if (!fits) continue;
    fit.fitting += n;
    const int value = t == kBlankTile ? 0 : c_tile_values[t];
    fit.max_value = max(fit.max_value, value);
  }
  return fit;
}

// BoardPlanes::encode's premium plane for a cell, or -1.
__device__ inline int premium_plane_offset(int cell) {
  const int lm = c_letter_mult[cell], wm = c_word_mult[cell];
  if (lm == 2) return 0;
  if (lm == 3) return 1;
  if (wm == 2) return 2;
  if (wm == 3) return 3;
  return -1;
}

// What a block shares while encoding one leaf.
struct LeafSmem {
  uint8_t sq[kSquares];
  int board_counts[kTileKinds];
  int seat_counts[3][kTileKinds];  // my leave, the opponent's, the pool
  int pool_size;
};

// Fills `s` from `leaf`, the whole block taking part.
__device__ void load_leaf(const LeafState& leaf, LeafSmem& s) {
  for (int t = threadIdx.x; t < kTileKinds; t += blockDim.x) s.board_counts[t] = 0;
  __syncthreads();
  for (int cell = threadIdx.x; cell < kSquares; cell += blockDim.x) {
    const uint8_t g = leaf.squares[cell];
    s.sq[cell] = g;
    if (g) atomicAdd(&s.board_counts[glyph_is_blank(g) ? kBlankTile : glyph_letter(g)], 1);
  }
  __syncthreads();
  if (threadIdx.x < kTileKinds) {
    const int t = threadIdx.x;
    s.seat_counts[0][t] = leaf.leaves[0][t];
    s.seat_counts[1][t] = leaf.leaves[1][t];
    s.seat_counts[2][t] =
      c_tile_distribution[t] - s.board_counts[t] - leaf.leaves[0][t] - leaf.leaves[1][t];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int pool = 0;
    for (int t = 0; t < kTileKinds; ++t) pool += s.seat_counts[2][t];
    s.pool_size = pool;
  }
  __syncthreads();
}

// The cross-check mask of `cell` for plays along `axis`: view 0 for
// horizontal plays, view 1 (indexed by its own row, the natural column) for
// vertical ones.
__device__ inline uint32_t cross_mask_at(const LeafState& leaf, int axis, int cell) {
  const int r = cell / kBoardSize, c = cell % kBoardSize;
  return axis == 0 ? leaf.cross_mask[0][cell] : leaf.cross_mask[1][c * kBoardSize + r];
}

// Whether `cell` has a tile beside it across the play axis.
__device__ inline bool has_cross_neighbor(const uint8_t* sq, int axis, int cell) {
  const int r = cell / kBoardSize, c = cell % kBoardSize;
  const int dr = axis == 0 ? 1 : 0, dc = 1 - dr;
  return (r - dr >= 0 && c - dc >= 0 && sq[(r - dr) * kBoardSize + (c - dc)] != 0) ||
         (r + dr < kBoardSize && c + dc < kBoardSize && sq[(r + dr) * kBoardSize + (c + dc)] != 0);
}

// Join plane `j` (seat-major, then axis, then fit / face value) at empty
// `cell`: set only on hook squares.
__device__ float join_plane_value(const LeafState& leaf, const LeafSmem& s, int j, int cell) {
  const int seat = j / 4, axis = (j / 2) % 2, kind = j % 2;
  if (!has_cross_neighbor(s.sq, axis, cell)) return 0.0f;
  const SeatFit fit = seat_fit(s.seat_counts[seat], cross_mask_at(leaf, axis, cell));
  if (fit.fitting == 0) return 0.0f;
  if (kind == 1) return float(fit.max_value) / kLeafFaceValueScale;
  return seat == 2 ? float(fit.fitting) / float(s.pool_size) : 1.0f;
}

// Spatial plane `plane` at `cell`: the board planes, then (on empty squares
// only) the cross-check and join planes.
__device__ float plane_value(const LeafState& leaf, const LeafSmem& s, int plane, int cell) {
  const uint8_t g = s.sq[cell];
  if (plane < kLeafBlankPlane) return (g != 0 && glyph_letter(g) == plane) ? 1.0f : 0.0f;
  if (plane == kLeafBlankPlane) return (g != 0 && glyph_is_blank(g)) ? 1.0f : 0.0f;
  if (plane < kLeafCrossPlane0) {
    return premium_plane_offset(cell) == plane - kLeafPremiumPlane0 ? 1.0f : 0.0f;
  }
  if (g != 0) return 0.0f;
  if (plane < kLeafJoinPlane0) {
    const int l = (plane - kLeafCrossPlane0) % 26, axis = (plane - kLeafCrossPlane0) / 26;
    return float((cross_mask_at(leaf, axis, cell) >> l) & 1u);
  }
  return join_plane_value(leaf, s, plane - kLeafJoinPlane0, cell);
}

// The row's scalars: the three seats' tile counts, then the score
// differential, both leave values and the pool size.
__device__ void write_leaf_scalars(const LeafState& leaf, const LeafSmem& s, float* sc) {
  if (threadIdx.x < kTileKinds) {
    for (int seat = 0; seat < 3; ++seat) {
      sc[seat * kTileKinds + threadIdx.x] = float(s.seat_counts[seat][threadIdx.x]);
    }
  }
  if (threadIdx.x == 0) {
    sc[kLeafScoreDiff] = float(leaf.scores[0] - leaf.scores[1]) / kLeafScoreScale;
    sc[kLeafScoreDiff + 1] = leave_value_of_counts(leaf.leaves[0]) / kLeafLeaveValueScale;
    sc[kLeafScoreDiff + 2] = leave_value_of_counts(leaf.leaves[1]) / kLeafLeaveValueScale;
    sc[kLeafScoreDiff + 3] = float(s.pool_size) / kLeafCountScale;
  }
}

__global__ void __launch_bounds__(kEncodeThreads)
  leaf_encode_kernel(const LeafState* leaves, const uint32_t* tasks, int count, float* spatial,
                     float* scalar) {
  const int b = blockIdx.x;
  if (b >= count) return;
  const LeafState& leaf = leaves[tasks[b]];
  __shared__ LeafSmem s;
  load_leaf(leaf, s);
  float* row = spatial + size_t(b) * kLeafSpatialPlanes * kSquares;
  for (int idx = threadIdx.x; idx < kLeafSpatialPlanes * kSquares; idx += blockDim.x) {
    row[idx] = plane_value(leaf, s, idx / kSquares, idx % kSquares);
  }
  write_leaf_scalars(leaf, s, scalar + size_t(b) * kLeafScalars);
}

template <typename T>
T* upload(const std::vector<T>& v, size_t* bytes) {
  T* d = nullptr;
  const size_t n = v.size() * sizeof(T);
  cuda_check(cudaMalloc(&d, n == 0 ? 1 : n), "malloc");
  if (n) cuda_check(cudaMemcpy(d, v.data(), n, cudaMemcpyHostToDevice), "memcpy");
  *bytes += n;
  return d;
}

// A device array that only grows.
template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t cap = 0;

  void reserve(size_t n) {
    if (n <= cap) return;
    cudaFree(data);
    cuda_check(cudaMalloc(&data, sizeof(T) * n), "malloc batch buffer");
    cap = n;
  }
  // Copies v to the device, growing as needed.
  void upload(const T* v, size_t n) {
    reserve(n == 0 ? 1 : n);
    if (n) cuda_check(cudaMemcpy(data, v, sizeof(T) * n, cudaMemcpyHostToDevice), "batch upload");
  }
  ~DeviceBuffer() { cudaFree(data); }
};

}  // namespace

struct DeviceTables {
  uint64_t* slots = nullptr;
  uint8_t* words16 = nullptr;
  uint64_t* blank_slots[2] = {nullptr, nullptr};
  uint32_t* blank_lists[2] = {nullptr, nullptr};  // 4 words per entry
  uint32_t* leave_keys = nullptr;
  float* leave_values = nullptr;
  uint32_t* arcs = nullptr;
  double* peg = nullptr;
  Scratch* scratch = nullptr;
  int* scratch_busy = nullptr;
  // The batch's inputs and outputs, grown to the largest batch so far.
  DeviceBuffer<RolloutTask> tasks;
  DeviceBuffer<PositionAbi> positions;
  DeviceBuffer<GroupAbi> groups;
  DeviceBuffer<OppLeaveAbi> opp_leaves;
  DeviceBuffer<RolloutResult> results;
  DeviceBuffer<LeafState> leaves;
  DeviceBuffer<uint32_t> leaf_tasks;  // the tasks encode_leaves was given
};

DeviceTables* upload_tables(const HostTables& host, size_t* bytes) {
  auto* d = new DeviceTables;
  *bytes = 0;
  d->slots = upload(host.slots, bytes);
  d->words16 = upload(host.words16, bytes);
  for (int b = 0; b < 2; ++b) {
    d->blank_slots[b] = upload(host.blank_slots[b], bytes);
    d->blank_lists[b] = upload(host.blank_lists[b], bytes);
  }
  d->leave_keys = upload(host.leave_keys, bytes);
  d->leave_values = upload(host.leave_values, bytes);
  d->arcs = upload(host.arcs, bytes);
  std::vector<double> peg(kPegTiles * kPegBags);
  for (int t = 0; t < kPegTiles; ++t)
    for (int b = 0; b < kPegBags; ++b) peg[size_t(t * kPegBags + b)] = host.peg[t][b];
  d->peg = upload(peg, bytes);
  cuda_check(cudaMalloc(&d->scratch, sizeof(Scratch) * kScratchSlots), "malloc scratch");
  cuda_check(cudaMalloc(&d->scratch_busy, sizeof(int) * kScratchSlots), "malloc scratch flags");
  cuda_check(cudaMemset(d->scratch_busy, 0, sizeof(int) * kScratchSlots), "clear scratch flags");

  Tables t;
  t.slots = d->slots;
  t.words16 = reinterpret_cast<const uint4*>(d->words16);
  for (int len = 0; len <= kMaxWordLen; ++len) {
    t.slot_offset[len] = host.slot_offset[len];
    t.slot_mask[len] = host.slot_mask[len];
    t.word_base[len] = host.word_base[len];
  }
  for (int b = 0; b < 2; ++b) {
    t.blank_slots[b] = d->blank_slots[b];
    t.blank_lists[b] = reinterpret_cast<const uint4*>(d->blank_lists[b]);
    for (int len = 0; len <= kMaxWordLen; ++len) {
      t.blank_slot_offset[b][len] = host.blank_slot_offset[b][len];
      t.blank_slot_mask[b][len] = host.blank_slot_mask[b][len];
    }
  }
  t.leave_keys = d->leave_keys;
  t.leave_values = d->leave_values;
  t.leave_mask = uint32_t(host.leave_keys.size() - 1);
  t.leave_shift = host.leave_shift;
  t.arcs = d->arcs;
  t.dawg_root = host.dawg_root;
  t.peg = d->peg;
  t.scratch = d->scratch;
  t.scratch_busy = d->scratch_busy;
  cuda_check(cudaMemcpyToSymbol(c_tables, &t, sizeof(t)), "tables symbol");
  cuda_check(cudaMemcpyToSymbol(c_letter_mult, host.letter_mult, sizeof(host.letter_mult)),
             "letter mult symbol");
  cuda_check(cudaMemcpyToSymbol(c_word_mult, host.word_mult, sizeof(host.word_mult)),
             "word mult symbol");
  cuda_check(cudaMemcpyToSymbol(c_tile_values, host.tile_values, sizeof(host.tile_values)),
             "tile values symbol");
  cuda_check(
    cudaMemcpyToSymbol(c_tile_distribution, host.tile_distribution, sizeof(host.tile_distribution)),
    "tile distribution symbol");
  cuda_check(cudaDeviceSynchronize(), "table upload");
  return d;
}

void free_tables(DeviceTables* d) {
  if (!d) return;
  cudaFree(d->slots);
  cudaFree(d->words16);
  for (int b = 0; b < 2; ++b) {
    cudaFree(d->blank_slots[b]);
    cudaFree(d->blank_lists[b]);
  }
  cudaFree(d->leave_keys);
  cudaFree(d->leave_values);
  cudaFree(d->arcs);
  cudaFree(d->peg);
  cudaFree(d->scratch);
  cudaFree(d->scratch_busy);
  delete d;
}

void run_rollouts(DeviceTables* d, const BatchAbi& batch, RolloutResult* results, double seconds[3],
                  unsigned long long phase_cycles[kPhases], unsigned long long counts[kCounts]) {
  using Clock = std::chrono::steady_clock;
  const auto since = [](Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  const size_t n = batch.tasks.size();
  d->results.reserve(n);
  d->leaves.reserve(n);
  const unsigned long long zero[kPhases] = {};
  cuda_check(cudaMemcpyToSymbol(g_phase_cycles, zero, sizeof(zero)), "phase counters");
  const unsigned long long zero_counts[kCounts] = {};
  cuda_check(cudaMemcpyToSymbol(g_counts, zero_counts, sizeof(zero_counts)), "work counters");
  auto t0 = Clock::now();
  d->tasks.upload(batch.tasks.data(), n);
  d->positions.upload(batch.positions.data(), batch.positions.size());
  d->groups.upload(batch.groups.data(), batch.groups.size());
  d->opp_leaves.upload(batch.opp_leaves.data(), batch.opp_leaves.size());
  seconds[0] = since(t0);
  t0 = Clock::now();
  if (n > 0) {
    rollout_kernel<<<unsigned(n), kThreads>>>(d->tasks.data, d->positions.data, d->groups.data,
                                              d->opp_leaves.data, d->results.data, d->leaves.data);
    cuda_check(cudaGetLastError(), "launch");
  }
  cuda_check(cudaDeviceSynchronize(), "kernel");
  seconds[1] = since(t0);
  t0 = Clock::now();
  if (n > 0) {
    cuda_check(
      cudaMemcpy(results, d->results.data, sizeof(RolloutResult) * n, cudaMemcpyDeviceToHost),
      "results download");
  }
  seconds[2] = since(t0);
  cuda_check(cudaMemcpyFromSymbol(phase_cycles, g_phase_cycles, sizeof(zero)), "phase readback");
  cuda_check(cudaMemcpyFromSymbol(counts, g_counts, sizeof(zero_counts)), "count readback");
}

const LeafState* device_leaves(const DeviceTables* d) { return d->leaves.data; }

void encode_leaves(DeviceTables* d, const uint32_t* tasks, int count, float* spatial,
                   float* scalar) {
  if (count <= 0) return;
  d->leaf_tasks.upload(tasks, size_t(count));
  leaf_encode_kernel<<<count, kEncodeThreads>>>(d->leaves.data, d->leaf_tasks.data, count, spatial,
                                                scalar);
  cuda_check(cudaGetLastError(), "leaf encoder launch");
  cuda_check(cudaDeviceSynchronize(), "leaf encoder");
}

void encode_leaves_to_host(DeviceTables* d, const uint32_t* tasks, int count, float* spatial,
                           float* scalar) {
  const size_t spatial_bytes = sizeof(float) * size_t(count) * kLeafSpatialPlanes * kSquares;
  const size_t scalar_bytes = sizeof(float) * size_t(count) * kLeafScalars;
  float *d_spatial = nullptr, *d_scalar = nullptr;
  cuda_check(cudaMalloc(&d_spatial, spatial_bytes), "malloc rows");
  cuda_check(cudaMalloc(&d_scalar, scalar_bytes), "malloc scalars");
  encode_leaves(d, tasks, count, d_spatial, d_scalar);
  cuda_check(cudaMemcpy(spatial, d_spatial, spatial_bytes, cudaMemcpyDeviceToHost),
             "rows download");
  cuda_check(cudaMemcpy(scalar, d_scalar, scalar_bytes, cudaMemcpyDeviceToHost),
             "scalars download");
  cudaFree(d_spatial);
  cudaFree(d_scalar);
}

}  // namespace scribblez::rollout_gpu
