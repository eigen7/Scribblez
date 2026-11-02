#pragma once

// Board's cross-checks on the device, shared by the kernels that compute
// them: the hasty-rollout kernel keeps a board's cross-checks up to date as
// its plies play, and the cross-check delta kernel finds the ones a candidate
// would change. Board::cross_check_at runs a warp per square; where a line
// here has a counterpart in board.cpp, the names match.
//
// A kernel reads its board through a `Squares` accessor, sq(view, r, c) being
// the glyph code (rollout_abi.h) at view square (r, c), so a kernel can lay a
// candidate's tiles over a board without copying it. View 1 is the board's
// transposed frame, as Board's cross-check arrays index it.

#include "gpu/rollout_abi.h"

#include <cstdint>

namespace scribblez::gpu {

constexpr uint32_t kAllLetters = (1u << 26) - 1u;
// A KWG arc: the next node in the low bits, then the end-of-siblings and
// accepts flags, and the tile in the top byte.
constexpr uint32_t kArcMask = 0x003fffffu;
constexpr uint32_t kIsEndBit = 0x00400000u;
constexpr uint32_t kAcceptsBit = 0x00800000u;

// The lexicon's KWG (Dictionary::arcs) in device memory.
struct DeviceDawg {
  const uint32_t* arcs;
  uint32_t root;
};

struct DStep {
  uint32_t next;
  bool valid;
  bool accepts;
};

__device__ inline DStep dict_step(const uint32_t* arcs, uint32_t node, uint8_t tile_value) {
  if (node == 0) return {0, false, false};
  for (uint32_t i = node;; ++i) {
    const uint32_t a = arcs[i];
    if ((a >> 24) == tile_value) return {a & kArcMask, true, (a & kAcceptsBit) != 0};
    if (a & kIsEndBit) return {0, false, false};
  }
}

// Rows top..bot of view column c: the filled run through (r, c), which is r
// alone when (r, c) has no vertical neighbor.
struct Run {
  int top;
  int bot;
};

template <class Squares>
__device__ inline Run perpendicular_run_bounds(const Squares& sq, int view, int r, int c) {
  Run run{r, r};
  while (run.top - 1 >= 0 && sq(view, run.top - 1, c) != 0) --run.top;
  while (run.bot + 1 < rollout_abi::kBoardSize && sq(view, run.bot + 1, c) != 0) ++run.bot;
  return run;
}

// Board::update_caches_after_place's squares: the view row of the empty
// square past end `end` (0 above, 1 below) of the perpendicular run through
// placed tile (vr, vc), or -1 off the board. The 4n of a play's n placed
// tiles are every square whose cross-check the play can change.
template <class Squares>
__device__ inline int run_end_row(const Squares& sq, int view, int vr, int vc, int end) {
  const Run run = perpendicular_run_bounds(sq, view, vr, vc);
  const int r = end == 0 ? run.top - 1 : run.bot + 1;
  return r >= 0 && r < rollout_abi::kBoardSize ? r : -1;
}

// Task t of those 4n run ends: view-major, then the placed tile, then the end.
struct RunEndTask {
  int view;
  int placed;
  int end;
};

__device__ inline RunEndTask run_end_task(int t, int n) {
  const int rest = t % (2 * n);
  return {t / (2 * n), rest / 2, rest % 2};
}

// Board::cross_check_letter_mask by a warp: lane L walks the run below with
// letter L placed, and a ballot collects the mask.
template <class Squares>
__device__ uint32_t cross_check_letter_mask(const Squares& sq, const uint32_t* arcs, int view,
                                            int c, uint32_t prefix_node, int r, int bot, int lane) {
  bool fits = false;
  if (lane < 26) {
    const DStep tr = dict_step(arcs, prefix_node, uint8_t(lane + 1));
    if (tr.valid) {
      bool acc = tr.accepts;
      uint32_t node = tr.next;
      bool ok = true;
      for (int rr = r + 1; rr <= bot; ++rr) {
        const DStep st =
          dict_step(arcs, node, uint8_t(rollout_abi::glyph_letter(sq(view, rr, c)) + 1));
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

// Board::CrossCheck.
struct CrossCheck {
  uint32_t mask = kAllLetters;
  int score = 0;
  bool has_neighbor = false;
};

// Board::cross_check_at by a warp, every lane receiving it. `value(g)` is the
// face value of glyph code g (0 for a blank).
template <class Squares, class TileValue>
__device__ CrossCheck cross_check_at(const Squares& sq, const TileValue& value, DeviceDawg dawg,
                                     int view, int r, int c, int lane) {
  CrossCheck cc;
  if (sq(view, r, c) != 0) return cc;  // filled squares: unused default
  const Run run = perpendicular_run_bounds(sq, view, r, c);
  cc.has_neighbor = run.top < r || run.bot > r;
  if (!cc.has_neighbor) return cc;

  // Walk the DAWG through the run above (r, c), totalling the face value of
  // the run above and below.
  uint32_t prefix_node = dawg.root;
  int prefix_score = 0;
  bool prefix_ok = true;
  for (int rr = run.top; rr <= r - 1; ++rr) {
    const uint8_t g = sq(view, rr, c);
    const DStep tr = dict_step(dawg.arcs, prefix_node, uint8_t(rollout_abi::glyph_letter(g) + 1));
    if (!tr.valid) {
      prefix_ok = false;
      break;
    }
    prefix_node = tr.next;
    prefix_score += value(g);
  }
  int suffix_score = 0;
  for (int rr = r + 1; rr <= run.bot; ++rr) suffix_score += value(sq(view, rr, c));
  cc.score = prefix_score + suffix_score;
  cc.mask =
    prefix_ok ? cross_check_letter_mask(sq, dawg.arcs, view, c, prefix_node, r, run.bot, lane) : 0;
  return cc;
}

}  // namespace scribblez::gpu
