#include "scribblez/game_state_encoder.h"

#include "scribblez/board_planes.h"
#include "scribblez/contingent_map.h"
#include "scribblez/glyph.h"
#include "scribblez/input_encoder.h"
#include "scribblez/tile.h"
#include "util/grid.h"

#include <cassert>
#include <cstring>

namespace scribblez {

namespace {

// Index into a single 15x15 plane: row-major if !flip, transposed if flip.
inline int plane_idx(int r, int c, bool flip) { return util::plane_index(r, c, kBoardSide, flip); }

// Placement plane: mark squares `m` placed tiles on. Uses Move::square_mask
// -- an absolute bitmask over the play's lane (bit k set iff lane cell k was a
// newly placed tile). No-op for EXCHANGE / PASS / game-start (leaving the plane
// all-zero).
void encode_placement_plane(const Move& m, bool flip, int plane, float* planes_out) {
  if (m.type() != MoveType::PLAY) return;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (r < 0 || r >= kBoardSide || c < 0 || c >= kBoardSide) break;
    planes_out[plane * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
  }
}

// Per-letter directional cross-check planes.
//
// Horizontal cross-check plane L marks empty squares where placing letter L
// would fuse with an existing left/right neighbor and pass horizontal-play
// cross-check constraints. Vertical cross-check planes are defined analogously
// using above/below neighbors and the transposed cross-check table.
void encode_cross_check_planes(const Board& board, bool flip, float* planes_out) {
  const auto& hcross = board.cross_checks(/*transposed=*/false);
  const auto& vcross = board.cross_checks(/*transposed=*/true);

  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      if (!board.at(r, c).is_empty()) continue;

      const bool fuse_horizontal = (c > 0 && !board.at(r, c - 1).is_empty()) ||
                                   (c + 1 < kBoardSide && !board.at(r, c + 1).is_empty());
      const bool fuse_vertical = (r > 0 && !board.at(r - 1, c).is_empty()) ||
                                 (r + 1 < kBoardSide && !board.at(r + 1, c).is_empty());

      const int out_ix = plane_idx(r, c, flip);
      if (fuse_horizontal) {
        // Horizontal play at (r,c):
        //   - perpendicular (vertical) legality comes from the non-transposed cache;
        //   - fused main-word (left/right) legality comes from the transposed cache.
        const uint32_t perp_mask = hcross[r * kBoardSide + c].mask;
        const uint32_t main_mask = vcross[c * kBoardSide + r].mask;
        const uint32_t mask = perp_mask & main_mask;
        for (int l = 0; l < 26; ++l) {
          if ((mask & (1u << l)) == 0) continue;
          planes_out[(kHorizontalCrossCheckPlane0 + l) * kBoardCells + out_ix] = 1.0f;
        }
      }

      if (fuse_vertical) {
        // Vertical play at (r,c):
        //   - fused main-word (above/below) legality comes from the non-transposed cache;
        //   - perpendicular (horizontal) legality comes from the transposed cache.
        const uint32_t main_mask = hcross[r * kBoardSide + c].mask;
        const uint32_t perp_mask = vcross[c * kBoardSide + r].mask;
        const uint32_t mask = main_mask & perp_mask;
        for (int l = 0; l < 26; ++l) {
          if ((mask & (1u << l)) == 0) continue;
          planes_out[(kVerticalCrossCheckPlane0 + l) * kBoardCells + out_ix] = 1.0f;
        }
      }
    }
  }
}

}  // namespace

// Every tile is in exactly one of (bag, board, p0 rack, p1 rack); the active
// player has no way to distinguish the bag from the opponent's rack, so
// unseen[i] = TILE_COUNTS[i] - (#tile i on board) - (#tile i in my_rack).
// This depends only on data the active player observes.
void compute_unseen_pool(uint8_t out[27], const Board& board, const Rack& my_rack) {
  for (int i = 0; i < 27; ++i) out[i] = static_cast<uint8_t>(TILE_COUNTS[i]);
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      Tile t = g.rack_tile();
      assert(out[t] > 0);
      --out[t];
    }
  }
  for (Tile t : my_rack.tiles()) {
    if (t.is_empty()) continue;
    assert(out[t] > 0);
    --out[t];
  }
}

namespace {

// Active player's rack as raw per-tile counts (27 floats).
void encode_rack_counts(const Rack& my_rack, float* out) {
  for (Tile t : my_rack.tiles()) {
    if (!t.is_empty()) out[t.index()] += 1.0f;
  }
}

// Unseen pool as a per-letter thermometer (100 floats): letter i owns a
// region of width TILE_COUNTS[i]; its first unseen[i] slots are 1.0, holes
// at the tail. Regions are concatenated in tile order (A..Z, blank).
void encode_unseen_pool_thermometer(const uint8_t unseen[27], float* out) {
  int offset = 0;
  for (int i = 0; i < 27; ++i) {
    for (int j = 0; j < unseen[i]; ++j) out[offset + j] = 1.0f;
    offset += TILE_COUNTS[i];
  }
  assert(offset == kUnseenPoolThermoFloats);
}

// Score differential as a thermometer (kScoreDiffThermoFloats floats): slot i is
// 1.0 iff the clipped diff >= (i - kScoreDiffClip).
void encode_score_diff_thermometer(int score_diff, float* out) {
  int clipped = score_diff;
  if (clipped < -kScoreDiffClip) clipped = -kScoreDiffClip;
  if (clipped > kScoreDiffClip) clipped = kScoreDiffClip;
  const int bin = clipped + kScoreDiffClip;  // 0..2*kScoreDiffClip
  for (int i = 0; i <= bin; ++i) out[i] = 1.0f;
}

// One move's metadata (kMoveMetaFloatsPerMove floats): move-type one-hot
// (indexed by MoveType) followed by num_glyphs.
void encode_move_meta(const Move& m, float* out) {
  out[static_cast<int>(m.type())] = 1.0f;
  out[kMoveMetaTypeFloats] = static_cast<float>(m.num_glyphs());
}

// All kScalarFloats scalar features. `self_move` / `opp_move` are the POV
// player's and opponent's most-recent moves, respectively. `score_diff` is
// the POV player's score advantage (score_active - score_opp).
void encode_scalars(const Rack& my_rack, const uint8_t unseen_pool[27], const Move& self_move,
                    const Move& opp_move, int score_diff, float* out) {
  encode_rack_counts(my_rack, out + kRackCountOffset);
  encode_unseen_pool_thermometer(unseen_pool, out + kUnseenPoolOffset);
  encode_score_diff_thermometer(score_diff, out + kScoreDiffOffset);
  encode_move_meta(self_move, out + kMoveMetaOffset);
  encode_move_meta(opp_move, out + kMoveMetaOffset + kMoveMetaFloatsPerMove);
}

// Shared back-end for both pre-move and post-PLAY encoding. Takes only
// POV-visible inputs; `score_diff` is the active player's score advantage.
void encode_pov(const Board& board, const Rack& my_rack, const Move& self_move,
                const Move& opp_move, int score_diff, const Dictionary& dict, bool apply_flip,
                float* out) {
  // The shared board-content block sits at the front of the spatial features,
  // with the placement/cross-check planes following it.
  static_assert(BoardPlanes::kBlankMarkerPlane == kBlankMarkerPlane &&
                  BoardPlanes::kPremiumPlane0 == kPremiumPlane0 &&
                  BoardPlanes::kPlanes == kSelfPlacementPlane,
                "post-move spatial layout must keep the shared board-plane block at the front");

  std::memset(out, 0, sizeof(float) * static_cast<size_t>(kInputFloats));
  BoardPlanes::encode(board, apply_flip, out);
  encode_placement_plane(self_move, apply_flip, kSelfPlacementPlane, out);
  encode_placement_plane(opp_move, apply_flip, kOppPlacementPlane, out);
  // The cross-check planes read the board's move-generation caches; build
  // them before encoding (a no-op when already valid) so they are
  // lexicon-accurate regardless of the board's prior cache state.
  board.ensure_movegen_caches(dict);
  encode_cross_check_planes(board, apply_flip, out);
  uint8_t unseen[27];
  compute_unseen_pool(unseen, board, my_rack);
  if (contingent_features_enabled()) {
    const ContingentMap contingent = ContingentMap::compute(board, my_rack, unseen, dict);
    contingent.encode_planes(apply_flip, out + kContingentPlane0 * kBoardCells);
    contingent.encode_scalars(out + kSpatialFloats + kContingentScalarOffset);
  }
  encode_scalars(my_rack, unseen, self_move, opp_move, score_diff, out + kSpatialFloats);
}

}  // namespace

namespace {
bool g_contingent_features_enabled = true;
}  // namespace

void set_contingent_features_enabled(bool enabled) { g_contingent_features_enabled = enabled; }
bool contingent_features_enabled() { return g_contingent_features_enabled; }

void GameStateEncoder::apply_move(const Move& move) {
  if (move.type() == MoveType::PLAY) {
    board_.apply(move);
    scores_[active_] += move.score();
  }
  // EXCHANGE / PASS: board and score are unchanged.
  last_move_by_[active_] = move;
  active_ = 1 - active_;
  ++turn_index_;
}

void GameStateEncoder::encode_input(int player, const Rack& my_rack, const Dictionary& dict,
                                    bool apply_flip, float* out) const {
  assert(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], dict, apply_flip, out);
}

void GameStateEncoder::encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                                    const Dictionary& dict, bool apply_flip,
                                                    float* out) const {
  assert(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(board_, my_rack, last_move_by_[player], last_move_by_[opp], score_diff, dict,
             apply_flip, out);
}

void GameStateEncoder::overwrite_score_diff(int score_diff, float* input_row) {
  float* block = input_row + kSpatialFloats + kScoreDiffOffset;
  std::memset(block, 0, sizeof(float) * static_cast<size_t>(kScoreDiffThermoFloats));
  encode_score_diff_thermometer(score_diff, block);
}

}  // namespace scribblez
