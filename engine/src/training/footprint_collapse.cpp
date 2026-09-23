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

// One anchored footprint's covered cells as flat plane indices (r*side + c),
// computed once per board and shared by the four heads.
struct CellList {
  uint8_t n = 0;
  std::array<uint16_t, kFootprintMaxK> cell{};
};

// footprint_cells for every anchored class on `board`. A class impossible on
// this board gets zero cells, so any probability it carries lands nowhere.
void compute_cells(const Board& board, std::vector<CellList>& cells) {
  cells.assign(kAnchoredFootprints, CellList{});
  std::array<std::pair<int, int>, kFootprintMaxK> rc;
  for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
    const int n = footprint_cells(cls, board, rc);
    CellList& cl = cells[cls];
    cl.n = uint8_t(n);
    for (int i = 0; i < n; ++i) cl.cell[i] = uint16_t(rc[i].first * kFootprintSide + rc[i].second);
  }
}

// Softmax over the classes `mask` keeps; illegal classes get zero. kPassClass
// is always legal, so the denominator is never zero.
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

// out[cell] = the summed probability of the anchored footprints covering cell.
// Pass and not-win place no tile, so they contribute to no cell.
void scatter(const std::vector<float>& prob, const std::vector<CellList>& cells, float* out) {
  std::fill_n(out, kBoardCells, 0.0f);
  for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
    const float p = prob[cls];
    if (p == 0.0f) continue;
    const CellList& cl = cells[cls];
    for (int i = 0; i < cl.n; ++i) out[cl.cell[i]] += p;
  }
}

// The four heads' masks, in kPlacementHeads order. A side's plays and win heads
// differ only at kExtraClass, and one opponent ply serves both sides: its mask
// is the opp heads' and its reach seeds the self heads' ply.
void fill_head_masks(const Board& board, const uint8_t* available_counts,
                     std::array<FootprintMask, kPlacementHeads>& masks) {
  const FootprintPly opp =
    footprint_ply(board, occupied_squares(board), kMaskTileBudget,
                  /*use_cross_checks=*/true, available_counts, /*win_head=*/false);
  const FootprintPly self =
    footprint_ply(board, opp.reach, kMaskTileBudget, /*use_cross_checks=*/false, nullptr,
                  /*win_head=*/false);
  masks[0] = opp.mask;
  masks[1] = self.mask;
  masks[2] = masks[0];
  masks[3] = masks[1];
  masks[2][kExtraClass] = true;  // opp_win opens the not-win class
  masks[3][kExtraClass] = true;  // self_win opens the not-win class
}

}  // namespace

void collapse_footprint_planes(const Board& board, const Dictionary& dict,
                               const uint8_t* available_counts, const float* raw, float* out) {
  board.ensure_movegen_caches(dict);
  std::array<FootprintMask, kPlacementHeads> masks;
  fill_head_masks(board, available_counts, masks);

  // A win head's extra not-win class carries no cells, so it enlarges the
  // softmax denominator (P[covers & win] <= P[covers]) without changing which
  // cells are covered.
  //
  // thread_local so repeated calls on a thread refill the ~44 KB cell table and
  // the probability buffer instead of reallocating them.
  thread_local std::vector<CellList> cells;
  thread_local std::vector<float> prob;
  compute_cells(board, cells);
  for (int h = 0; h < kPlacementHeads; ++h) {
    masked_softmax(raw + size_t(h) * kFootprintClasses, masks[h], prob);
    scatter(prob, cells, out + size_t(h) * kBoardCells);
  }
}

void masked_placement_distributions(const Board& board, const Dictionary& dict,
                                    const uint8_t* available_counts, const float* raw, float* out) {
  board.ensure_movegen_caches(dict);
  std::array<FootprintMask, kPlacementHeads> masks;
  fill_head_masks(board, available_counts, masks);

  thread_local std::vector<float> prob;
  for (int h = 0; h < kPlacementHeads; ++h) {
    masked_softmax(raw + size_t(h) * kFootprintClasses, masks[h], prob);
    std::copy_n(prob.data(), kFootprintClasses, out + size_t(h) * kFootprintClasses);
  }
}

void collapse_footprint_legal_cells(const Board& board, const Dictionary& dict,
                                    const uint8_t* available_counts, float* out) {
  board.ensure_movegen_caches(dict);
  std::array<FootprintMask, kPlacementHeads> masks;
  fill_head_masks(board, available_counts, masks);

  std::vector<CellList> cells;
  compute_cells(board, cells);
  std::fill_n(out, size_t(kPlacementHeads) * kBoardCells, 0.0f);
  for (int h = 0; h < kPlacementHeads; ++h) {
    float* head_out = out + size_t(h) * kBoardCells;
    for (int cls = 0; cls < kAnchoredFootprints; ++cls) {
      if (!masks[h][cls]) continue;
      const CellList& cl = cells[cls];
      for (int i = 0; i < cl.n; ++i) head_out[cl.cell[i]] = 1.0f;
    }
  }
}

}  // namespace scribblez
