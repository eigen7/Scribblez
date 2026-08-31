#include "training/footprint_mask.h"

#include "util/math.h"

#include <array>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

// Mark the legal footprint classes given a multi-tile per-cell predicate
// `cell_ok(board_horizontal, r, c)`, a lone-tile predicate `lone_ok(r, c)`, and
// the tile-count cap `kmax`. Shared by the opp and self masks, which differ only
// in those predicates and the cap.
template <typename CellOk, typename LoneOk>
void mark_footprints(const Board& board, int kmax, bool flip, bool win_head, FootprintMask& mask,
                     CellOk cell_ok, LoneOk lone_ok) {
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
        if (lone_ok(anchor_r, anchor_c)) mask[cls] = true;
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

// The opponent's tile availability as a bit-set for a fast per-cell test: bit L
// set iff at least one of tile L is in stock, indexed like the 27-count array
// (0..25 = A..Z, 26 = blank).
using tile_set_t = uint32_t;
inline constexpr tile_set_t kBlankTile = 1u << 26;                     // wildcard blank
inline constexpr tile_set_t kAllTiles = kAllLettersMask | kBlankTile;  // every tile in stock

// The available tiles distilled from a 27-count array. A null array is
// "everything in stock", which collapses the availability test to board legality.
tile_set_t available_tiles(const uint8_t* counts) {
  if (counts == nullptr) return kAllTiles;
  tile_set_t avail = 0;
  for (int t = 0; t < 27; ++t)
    if (counts[t] > 0) avail |= (1u << t);
  return avail;
}

// Can some AVAILABLE letter play at empty cell (r,c) in board-frame orientation
// `horizontal`? Its perpendicular cross-check mask must share a letter with
// `avail` (a blank is a wildcard). Cache indexing matches the input encoder: a
// horizontal word's cross-words run down the columns (non-transposed [r*side+c]),
// a vertical word's along the rows (transposed [c*side+r]).
bool cell_admits_letter(const Board& board, tile_set_t avail, bool horizontal, int r, int c) {
  const CrossCheck& cc = horizontal ? board.cross_checks(false)[r * BOARD_SIZE + c]
                                    : board.cross_checks(true)[c * BOARD_SIZE + r];
  if (avail & kBlankTile) return cc.mask != 0;  // a wildcard plays wherever any letter is legal
  return (cc.mask & avail) != 0;                // an available letter that is legal here
}

// Can some AVAILABLE letter play as a LONE tile at (r,c)? It forms both its
// cross-words at once, so its letter must clear both cross-checks -- their mask
// intersection, not cell_admits_letter's per-axis test.
bool lone_tile_admits_letter(const Board& board, tile_set_t avail, int r, int c) {
  const CrossCheck& vert = board.cross_checks(false)[r * BOARD_SIZE + c];
  const CrossCheck& horiz = board.cross_checks(true)[c * BOARD_SIZE + r];
  const uint32_t allowed = vert.mask & horiz.mask;  // letters legal in both words the tile forms
  if (avail & kBlankTile) return allowed != 0;      // a wildcard fills any jointly-legal square
  return (allowed & avail) != 0;                    // a jointly-legal letter that is in stock
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

void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool flip, bool win_head, FootprintMask& mask) {
  const int kmax = tile_budget < kFootprintMaxK ? tile_budget : kFootprintMaxK;
  const tile_set_t avail = available_tiles(available_counts);
  mark_footprints(
    board, kmax, flip, win_head, mask,
    [&](bool horizontal, int r, int c) {
      return cell_admits_letter(board, avail, horizontal, r, c);
    },
    [&](int r, int c) { return lone_tile_admits_letter(board, avail, r, c); });
}

void self_footprint_mask(const Board& board, int self_budget, int opp_budget, bool flip,
                         bool win_head, FootprintMask& mask) {
  const std::array<int, kFootprintCells> dist = tiles_to_reach(board);
  const int reach = opp_budget + self_budget;  // opp bridges, then the mover finishes
  const int kmax = self_budget < kFootprintMaxK ? self_budget : kFootprintMaxK;
  // Orientation-free reachability -- the self mask is cross-check-oblivious, so
  // the lone-tile test coincides with the multi-tile per-cell one.
  const auto reachable = [&](int r, int c) { return dist[r * kFootprintSide + c] <= reach; };
  mark_footprints(
    board, kmax, flip, win_head, mask, [&](bool, int r, int c) { return reachable(r, c); },
    reachable);
}

}  // namespace scribblez
