#include "training/footprint_collapse.h"

#include "game/tile.h"
#include "training/footprint_mask.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

inline constexpr int kBoardCells = kFootprintSide * kFootprintSide;

// One anchored footprint's covered cells as flat plane indices (r*side + c, in
// the flip frame footprint_cells reports), precomputed once per board and reused
// across the four heads.
struct CellList {
  uint8_t n = 0;
  std::array<uint16_t, kFootprintMaxK> cell{};
};

// footprint_cells for every anchored class on `board`, into `cells`. An
// impossible class on this board reports zero cells and simply never receives
// probability.
void compute_cells(const Board& board, bool flip, std::vector<CellList>& cells) {
  cells.assign(kAnchoredFootprints, CellList{});
  std::array<std::pair<int, int>, kFootprintMaxK> rc;
  for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
    const int n = footprint_cells(cls, board, flip, rc);
    CellList& cl = cells[cls];
    cl.n = uint8_t(n);
    for (int i = 0; i < n; ++i) cl.cell[i] = uint16_t(rc[i].first * kFootprintSide + rc[i].second);
  }
}

// Softmax of `logits` over the classes `mask` keeps, into `prob` (illegal
// classes get zero). kPassClass is always legal, so the denominator is never
// zero.
void masked_softmax(const float* logits, const FootprintMask& mask, std::vector<float>& prob) {
  prob.assign(kFootprintClasses, 0.0f);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int c = 0; c < kFootprintClasses; ++c)
    if (mask[c]) max_logit = std::max(max_logit, logits[c]);
  float sum = 0.0f;
  for (int c = 0; c < kFootprintClasses; ++c) {
    if (!mask[c]) continue;
    const float e = std::exp(logits[c] - max_logit);
    prob[c] = e;
    sum += e;
  }
  const float inv = 1.0f / sum;
  for (int c = 0; c < kFootprintClasses; ++c) prob[c] *= inv;
}

// out[cell] = sum over anchored footprints of prob[footprint] on each cell it
// covers. pass / not-win carry no cells, so they drop out of the per-cell
// marginal (they placed no tile), exactly as the old occupancy plane.
void scatter(const std::vector<float>& prob, const std::vector<CellList>& cells, float* out) {
  std::fill_n(out, kBoardCells, 0.0f);
  for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
    const float p = prob[cls];
    if (p == 0.0f) continue;
    const CellList& cl = cells[cls];
    for (int i = 0; i < cl.n; ++i) out[cl.cell[i]] += p;
  }
}

}  // namespace

void collapse_footprint_planes(const Board& board, const Dictionary& dict,
                               const uint8_t* available_counts, bool flip, const float* raw,
                               float* out) {
  board.ensure_movegen_caches(dict);

  // The four heads' legality: opp / self, each with a plays (win_head=false) and
  // a win (win_head=true) variant. win_head toggles only kExtraClass (the
  // not-win outcome), which carries no cells -- so it changes the softmax
  // denominator (P[covers & win] <= P[covers]) but not which cells are covered.
  // Availability (`available_counts`) gates the opp heads only; the self heads
  // never take it (see footprint_mask.h).
  FootprintMask opp_mask, self_mask;
  opp_footprint_mask(board, available_counts, kMaskTileBudget, flip, /*win_head=*/false, opp_mask);
  self_footprint_mask(board, kMaskTileBudget, kMaskTileBudget, flip, /*win_head=*/false, self_mask);
  FootprintMask opp_win_mask = opp_mask, self_win_mask = self_mask;
  opp_win_mask[kExtraClass] = true;
  self_win_mask[kExtraClass] = true;
  const std::array<const FootprintMask*, kPlacementHeads> masks = {&opp_mask, &self_mask,
                                                                   &opp_win_mask, &self_win_mask};

  // Reused across calls on this thread -- the generator collapses many
  // candidates per thread, so the ~44KB cells buffer and the prob buffer are
  // allocated once and refilled, not per candidate.
  thread_local std::vector<CellList> cells;
  thread_local std::vector<float> prob;
  compute_cells(board, flip, cells);
  for (int h = 0; h < kPlacementHeads; ++h) {
    masked_softmax(raw + size_t(h) * kFootprintClasses, *masks[h], prob);
    scatter(prob, cells, out + size_t(h) * kBoardCells);
  }
}

}  // namespace scribblez
