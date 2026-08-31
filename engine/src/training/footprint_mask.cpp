#include "training/footprint_mask.h"

#include "util/math.h"

#include <array>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// Mark the legal footprint classes given a per-cell placeability predicate
// `cell_ok(board_horizontal, r, c)` and the tile-count cap `kmax`. Shared by the
// opp and self masks, which differ only in that predicate and cap. A class
// (anchor, orientation, k) is legal iff its anchor is empty, its first k empty
// cells stay on the board, and each passes the predicate for the play's
// board-frame orientation. A lone tile (k==1) is orientation-free -- legal if it
// passes the predicate along either axis.
template <typename CellOk>
void mark_footprints(const Board& board, int kmax, bool flip, bool win_head, CellOk cell_ok,
                     FootprintMask& mask) {
  mask.fill(false);
  for (int cell = 0; cell < kFootprintCells; ++cell) {
    int anchor_r = cell / kFootprintSide;
    int anchor_c = cell % kFootprintSide;
    if (flip) std::swap(anchor_r, anchor_c);  // un-flip the anchor into board coords
    if (!board.at(anchor_r, anchor_c).is_empty()) continue;

    for (int slot = 0; slot < kSlotsPerCell; ++slot) {
      bool horizontal;
      int k;
      footprint_slot_decode(slot, horizontal, k);  // orientation in the flip frame
      if (k > kmax) continue;
      const int cls = cell * kSlotsPerCell + slot;

      if (k == 1) {
        if (cell_ok(true, anchor_r, anchor_c) || cell_ok(false, anchor_r, anchor_c)) {
          mask[cls] = true;
        }
        continue;
      }

      const bool board_horizontal = flip ? !horizontal : horizontal;
      int count = 0;
      int r = anchor_r;
      int c = anchor_c;
      bool ok = true;
      while (r < kFootprintSide && c < kFootprintSide && count < k) {
        if (board.at(r, c).is_empty()) {
          if (!cell_ok(board_horizontal, r, c)) {
            ok = false;
            break;
          }
          ++count;
        }
        if (board_horizontal) {
          ++c;
        } else {
          ++r;
        }
      }
      if (ok && count == k) mask[cls] = true;  // count<k means the edge cut it short
    }
  }
  mask[kPassClass] = true;
  mask[kExtraClass] = win_head;
}

// The opponent's tile availability, distilled from the 27-count supply for a
// fast per-cell test: `letters` has bit L set iff at least one of letter L is in
// stock, and `blank` iff a wildcard blank is. A null supply is "everything in
// stock", which makes the availability test collapse to pure board legality.
struct Supply {
  uint32_t letters = kAllLettersMask;  // null-supply default: every letter available
  bool blank = true;
};

Supply make_supply(const uint8_t* counts) {
  if (counts == nullptr) return Supply{};
  Supply s{0u, counts[26] > 0};
  for (int l = 0; l < 26; ++l)
    if (counts[l] > 0) s.letters |= (1u << l);
  return s;
}

// Can some AVAILABLE letter play at board cell (r,c) as part of a word in
// board-frame orientation `horizontal`? The perpendicular cross-check picks the
// legal letters (non-empty, or all letters if the square is unconstrained); the
// letter must also be in `supply`. A blank in `supply` is a wildcard, so it
// satisfies any board-legal square and the test reduces to board legality. A
// horizontal word's cross-words run down the columns (non-transposed cache,
// indexed [r*side+c]); a vertical word's along the rows (transposed cache,
// [c*side+r]) -- matching the input encoder.
bool cell_admits_letter(const Board& board, const Supply& supply, bool horizontal, int r, int c) {
  const CrossCheck& cc = horizontal ? board.cross_checks(false)[r * BOARD_SIZE + c]
                                    : board.cross_checks(true)[c * BOARD_SIZE + r];
  if (supply.blank) return !cc.has_neighbor || cc.mask != 0;  // wildcard fills any legal square
  if (!cc.has_neighbor) return supply.letters != 0;           // unconstrained: any available tile
  return (cc.mask & supply.letters) != 0;                     // legal AND in stock
}

// Tiles-to-reach distance field: d[Z] = the fewest tiles that must be placed to
// bridge to empty cell Z from the current structure. Multi-source 4-neighbour
// BFS with every occupied cell a 0 seed (words can turn at any tile), so it is a
// cross-check-oblivious lower bound on the true tile cost -- hence a sound
// over-approximation of reachability. A tile-less board (game start) leaves the
// field at 0 everywhere so nothing is spuriously masked.
std::array<int, kFootprintCells> tiles_to_reach(const Board& board) {
  std::array<int, kFootprintCells> dist;
  dist.fill(INT_MAX);
  if (board.num_tiles() == 0) {
    dist.fill(0);
    return dist;
  }
  std::vector<int> q;
  for (int i = 0; i < kFootprintCells; ++i) {
    if (!board.at(i / kFootprintSide, i % kFootprintSide).is_empty()) {
      dist[i] = 0;
      q.push_back(i);
    }
  }
  for (size_t head = 0; head < q.size(); ++head) {
    const int cur = q[head];
    const int r = cur / kFootprintSide;
    const int c = cur % kFootprintSide;
    for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
      const int nr = r + dr;
      const int nc = c + dc;
      if (nr < 0 || nr >= kFootprintSide || nc < 0 || nc >= kFootprintSide) continue;
      const int ni = nr * kFootprintSide + nc;
      if (dist[ni] != INT_MAX) continue;           // seeded (occupied) or already reached
      if (!board.at(nr, nc).is_empty()) continue;  // occupied cells stay 0 seeds
      dist[ni] = dist[cur] + 1;
      q.push_back(ni);
    }
  }
  return dist;
}

}  // namespace

void opp_footprint_mask(const Board& board, const uint8_t* supply, int tile_budget, bool flip,
                        bool win_head, FootprintMask& mask) {
  const int kmax = tile_budget < kFootprintMaxK ? tile_budget : kFootprintMaxK;
  const Supply s = make_supply(supply);
  mark_footprints(
    board, kmax, flip, win_head,
    [&](bool horizontal, int r, int c) { return cell_admits_letter(board, s, horizontal, r, c); },
    mask);
}

void self_footprint_mask(const Board& board, int self_budget, int opp_budget, bool flip,
                         bool win_head, FootprintMask& mask) {
  const std::array<int, kFootprintCells> dist = tiles_to_reach(board);
  const int reach = opp_budget + self_budget;  // opp bridges, then the mover finishes
  const int kmax = self_budget < kFootprintMaxK ? self_budget : kFootprintMaxK;
  mark_footprints(
    board, kmax, flip, win_head,
    [&](bool /*orientation-free*/, int r, int c) { return dist[r * kFootprintSide + c] <= reach; },
    mask);
}

}  // namespace scribblez
