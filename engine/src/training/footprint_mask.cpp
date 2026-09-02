#include "training/footprint_mask.h"

#include "util/math.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace scribblez {

namespace {

// The mover's tile availability as a bit-set for a fast per-cell test: bit L
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

// An empty cell (r,c) is orthogonally adjacent to a seed cell.
bool touches_seed(const Reach& seed, int r, int c) {
  for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
    const int nr = r + dr;
    const int nc = c + dc;
    if (nr < 0 || nr >= kFootprintSide || nc < 0 || nc >= kFootprintSide) continue;
    if (seed.reachable(nr, nc)) return true;
  }
  return false;
}

// Record a legal k-tile footprint's covered cells onto `reach`: a cell not in the
// seed takes the fewest tiles of any footprint covering it; a seed cell keeps
// its seed depth.
void cover(const Reach& seed, const int* cells, int n, int k, Reach& reach) {
  for (int i = 0; i < n; ++i) {
    const int idx = cells[i];
    if (!seed.reachable(idx)) reach.tiles[idx] = std::min<uint8_t>(reach.tiles[idx], uint8_t(k));
  }
}

}  // namespace

bool Reach::empty() const {
  return std::all_of(tiles.begin(), tiles.end(), [](uint8_t t) { return t == kUnreachable; });
}

Reach occupied_reach(const Board& board) {
  Reach s;
  s.tiles.fill(Reach::kUnreachable);
  for (int i = 0; i < kFootprintCells; ++i)
    if (!board.at(i / kFootprintSide, i % kFootprintSide).is_empty()) s.tiles[i] = 0;
  return s;
}

Expansion expand(const Board& board, const Reach& seed, int budget, bool use_cross_checks,
                 const uint8_t* available_counts, bool win_head) {
  Expansion out;
  out.mask.fill(false);
  out.reach = seed;
  const int kmax = std::min(budget, kFootprintMaxK);
  const tile_set_t avail = use_cross_checks ? available_tiles(available_counts) : kAllTiles;
  const bool adjacency_gate = !seed.empty();  // nothing to abut on the opener's empty board
  // TODO(perf): most anchors abut nothing and get masked, yet we scan all
  // kFootprintCells of them; crawling footprints out from the seed would touch
  // only the connected few (~222 vs 2925 for a centred opener). Mask-build has
  // not shown up in profiling, so it is deferred.
  std::array<int, kFootprintMaxK> covered;
  for (int cell = 0; cell < kFootprintCells; ++cell) {
    const int anchor_r = cell / kFootprintSide;
    const int anchor_c = cell % kFootprintSide;
    if (!board.at(anchor_r, anchor_c).is_empty()) continue;

    for (int slot = 0; slot < kSlotsPerCell; ++slot) {
      bool horizontal;
      int k;
      footprint_slot_decode(slot, horizontal, k);
      if (k > kmax) continue;
      const int cls = cell * kSlotsPerCell + slot;

      if (k == 1) {
        const bool ok =
          (!use_cross_checks || lone_tile_admits_letter(board, avail, anchor_r, anchor_c)) &&
          (!adjacency_gate || touches_seed(seed, anchor_r, anchor_c));
        if (ok) {
          out.mask[cls] = true;
          cover(seed, &cell, 1, 1, out.reach);
        }
        continue;
      }

      int count = 0;
      int r = anchor_r;
      int c = anchor_c;
      bool ok = true;
      bool connected = !adjacency_gate;  // vacuously satisfied when the gate is off
      while (r < kFootprintSide && c < kFootprintSide && count < k) {
        if (board.at(r, c).is_empty()) {
          if (use_cross_checks && !cell_admits_letter(board, avail, horizontal, r, c)) {
            ok = false;
            break;
          }
          if (!connected && touches_seed(seed, r, c)) connected = true;
          covered[count++] = r * kFootprintSide + c;
        }
        if (horizontal) {
          ++c;
        } else {
          ++r;
        }
      }
      // count<k means the edge cut it short; !connected means it floats free of
      // the seed (a disconnected placement).
      if (ok && count == k && connected) {
        out.mask[cls] = true;
        cover(seed, covered.data(), k, k, out.reach);
      }
    }
  }
  out.mask[kPassClass] = true;
  out.mask[kExtraClass] = win_head;
  return out;
}

void opp_footprint_mask(const Board& board, const uint8_t* available_counts, int tile_budget,
                        bool win_head, FootprintMask& mask) {
  mask = expand(board, occupied_reach(board), tile_budget, /*use_cross_checks=*/true,
                available_counts, win_head)
           .mask;
}

void self_footprint_mask(const Board& board, int self_budget, int opp_budget,
                         const uint8_t* opp_available_counts, bool win_head, FootprintMask& mask) {
  const Expansion opp = expand(board, occupied_reach(board), opp_budget, /*use_cross_checks=*/true,
                               opp_available_counts, /*win_head=*/false);
  mask = expand(board, opp.reach, self_budget, /*use_cross_checks=*/false, nullptr, win_head).mask;
}

void footprint_reachable_cells(const Board& board, const uint8_t* available_counts, int tile_budget,
                               float* out) {
  const Reach reach = expand(board, occupied_reach(board), tile_budget, /*use_cross_checks=*/true,
                             available_counts, /*win_head=*/false)
                        .reach;
  // A covered cell carries a depth of 1..kFootprintMaxK; the occupied seed sits
  // at 0 and is never covered.
  for (int i = 0; i < kFootprintCells; ++i) {
    const uint8_t t = reach.tiles[i];
    out[i] = (t != Reach::kUnreachable && t > 0) ? 1.0f : 0.0f;
  }
}

}  // namespace scribblez
