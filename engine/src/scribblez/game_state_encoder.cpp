#include "scribblez/game_state_encoder.h"

#include "scribblez/board_planes.h"
#include "scribblez/contingent_map.h"
#include "scribblez/glyph.h"
#include "scribblez/input_encoder.h"
#include "scribblez/tile.h"
#include "util/grid.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace scribblez {

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

// Index into a single 15x15 plane: row-major if !flip, transposed if flip.
inline int plane_idx(int r, int c, bool flip) { return util::plane_index(r, c, kBoardSide, flip); }

// Everything the block writers read: the POV-visible position state plus the
// spec and the (optional) precomputed contingent map, which two blocks share.
struct PovCtx {
  const Board& board;
  const Rack& my_rack;
  const Move& self_move;
  const Move& opp_move;
  int score_diff;
  const uint8_t* unseen;  // 27 per-kind unseen-pool counts
  bool apply_flip;
  const ContingentMap* contingent;  // null iff the spec excludes the blocks
  const Rack* opp_rack;             // null iff the spec excludes the open-rack block
};

// Placement plane (1 plane at `out`): mark squares `m` placed tiles on. Uses
// Move::square_mask -- an absolute bitmask over the play's lane (bit k set iff
// lane cell k was a newly placed tile). No-op for EXCHANGE / PASS / game-start
// (leaving the plane all-zero).
int encode_placement_plane(const Move& m, bool flip, float* out) {
  if (m.type() != MoveType::PLAY) return 1;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    if (r < 0 || r >= kBoardSide || c < 0 || c >= kBoardSide) break;
    out[plane_idx(r, c, flip)] = 1.0f;
  }
  return 1;
}

// Per-letter directional cross-check planes (kCrossCheckPlanes planes at
// `out`: horizontal A..Z, then vertical A..Z).
//
// Horizontal cross-check plane L marks empty squares where placing letter L
// would fuse with an existing left/right neighbor and pass horizontal-play
// cross-check constraints. Vertical cross-check planes are defined analogously
// using above/below neighbors and the transposed cross-check table.
int encode_cross_check_planes(const Board& board, bool flip, float* out) {
  float* h_planes = out;
  float* v_planes = out + kHorizontalCrossCheckPlanes * kBoardCells;
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
          h_planes[l * kBoardCells + out_ix] = 1.0f;
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
          v_planes[l * kBoardCells + out_ix] = 1.0f;
        }
      }
    }
  }
  return kCrossCheckPlanes;
}

// Active player's rack as raw per-tile counts (27 floats).
int encode_rack_counts(const Rack& my_rack, float* out) {
  for (Tile t : my_rack.tiles()) {
    if (!t.is_empty()) out[t.index()] += 1.0f;
  }
  return kRackCountFloats;
}

// Unseen pool as a per-letter thermometer (100 floats): letter i owns a
// region of width TILE_COUNTS[i]; its first unseen[i] slots are 1.0, holes
// at the tail. Regions are concatenated in tile order (A..Z, blank).
int encode_unseen_pool_thermometer(const uint8_t unseen[27], float* out) {
  int offset = 0;
  for (int i = 0; i < 27; ++i) {
    for (int j = 0; j < unseen[i]; ++j) out[offset + j] = 1.0f;
    offset += TILE_COUNTS[i];
  }
  assert(offset == kUnseenPoolThermoFloats);
  return kUnseenPoolThermoFloats;
}

// Score differential as a thermometer (kScoreDiffThermoFloats floats): slot i is
// 1.0 iff the clipped diff >= (i - kScoreDiffClip).
int encode_score_diff_thermometer(int score_diff, float* out) {
  int clipped = score_diff;
  if (clipped < -kScoreDiffClip) clipped = -kScoreDiffClip;
  if (clipped > kScoreDiffClip) clipped = kScoreDiffClip;
  const int bin = clipped + kScoreDiffClip;  // 0..2*kScoreDiffClip
  for (int i = 0; i <= bin; ++i) out[i] = 1.0f;
  return kScoreDiffThermoFloats;
}

// Last-2-move metadata (kMoveMetaFloats floats): for the POV player's and then
// the opponent's most recent move, a move-type one-hot (indexed by MoveType)
// followed by num_glyphs.
int encode_move_meta(const Move& self_move, const Move& opp_move, float* out) {
  out[static_cast<int>(self_move.type())] = 1.0f;
  out[kMoveMetaTypeFloats] = static_cast<float>(self_move.num_glyphs());
  float* opp = out + kMoveMetaFloatsPerMove;
  opp[static_cast<int>(opp_move.type())] = 1.0f;
  opp[kMoveMetaTypeFloats] = static_cast<float>(opp_move.num_glyphs());
  return kMoveMetaFloats;
}

// Registry dispatch: write one block at `out`, returning its plane count.
int encode_spatial_block(SpatialBlockId id, const PovCtx& ctx, float* out) {
  switch (id) {
    case SpatialBlockId::kBoard:
      static_assert(BoardPlanes::kPlanes == kBoardBlockPlanes,
                    "the registry's board block must match the shared board-plane encoder");
      BoardPlanes::encode(ctx.board, ctx.apply_flip, out);
      return BoardPlanes::kPlanes;
    case SpatialBlockId::kSelfPlacement:
      return encode_placement_plane(ctx.self_move, ctx.apply_flip, out);
    case SpatialBlockId::kOppPlacement:
      return encode_placement_plane(ctx.opp_move, ctx.apply_flip, out);
    case SpatialBlockId::kCrossChecks:
      return encode_cross_check_planes(ctx.board, ctx.apply_flip, out);
    case SpatialBlockId::kContingent:
      ctx.contingent->encode_planes(ctx.apply_flip, out);
      return kContingentPlanes;
  }
  std::abort();  // unreachable: the switch covers every SpatialBlockId
}

// Registry dispatch: write one block at `out`, returning its float count.
int encode_scalar_block(ScalarBlockId id, const PovCtx& ctx, float* out) {
  switch (id) {
    case ScalarBlockId::kRackCounts:
      return encode_rack_counts(ctx.my_rack, out);
    case ScalarBlockId::kUnseenPool:
      return encode_unseen_pool_thermometer(ctx.unseen, out);
    case ScalarBlockId::kScoreDiff:
      return encode_score_diff_thermometer(ctx.score_diff, out);
    case ScalarBlockId::kMoveMeta:
      return encode_move_meta(ctx.self_move, ctx.opp_move, out);
    case ScalarBlockId::kContingent:
      ctx.contingent->encode_scalars(out);
      return kContingentScalarFloats;
    case ScalarBlockId::kOppRackCounts:
      return encode_rack_counts(*ctx.opp_rack, out);
  }
  std::abort();  // unreachable: the switch covers every ScalarBlockId
}

// Always-on layout guard (kept in release builds -- the checks are a handful
// of integer compares per row): a block writer disagreeing with the registry,
// or the walk not landing exactly on the spec's totals, is memory corruption
// in the making.
void check_layout(bool ok, const char* what) {
  if (ok) return;
  std::fprintf(stderr, "input encode: %s disagrees with the layout registry\n", what);
  std::abort();
}

// Shared back-end for both pre-move and post-PLAY encoding. Takes only
// POV-visible inputs; `score_diff` is the active player's score advantage.
// Writes the spec's blocks in registry order, validating every block's size
// and the final totals against the registry.
void encode_pov(const InputEncodingSpec& spec, const Board& board, const Rack& my_rack,
                const Move& self_move, const Move& opp_move, int score_diff, const Rack* opp_rack,
                bool apply_flip, float* out) {
  check_layout(!spec.opp_rack_input || opp_rack != nullptr,
               "an open-rack spec encoded without the opponent rack");
  std::memset(out, 0, sizeof(float) * static_cast<size_t>(input_floats(spec)));
  // The cross-check and contingent blocks read the board's move-generation
  // caches; build them up front (a no-op when already valid) so they are
  // lexicon-accurate regardless of the board's prior cache state.
  board.ensure_movegen_caches(*spec.dict);
  uint8_t unseen[27];
  compute_unseen_pool(unseen, board, my_rack);
  std::optional<ContingentMap> contingent;
  if (spec.contingent_features) {
    contingent = ContingentMap::compute(board, my_rack, unseen, *spec.dict);
  }
  const PovCtx ctx{board,
                   my_rack,
                   self_move,
                   opp_move,
                   score_diff,
                   unseen,
                   apply_flip,
                   contingent ? &*contingent : nullptr,
                   spec.opp_rack_input ? opp_rack : nullptr};

  float* cursor = out;
  for (const SpatialBlockDef& def : kSpatialBlocks) {
    if (def.contingent_only && !spec.contingent_features) continue;
    const int planes = encode_spatial_block(def.id, ctx, cursor);
    check_layout(planes == def.planes, "a spatial block's plane count");
    cursor += planes * kBoardCells;
  }
  check_layout(cursor == out + spatial_floats(spec), "the spatial section's total");
  for (const ScalarBlockDef& def : kScalarBlocks) {
    if (!scalar_block_included(def, spec)) continue;
    const int floats = encode_scalar_block(def.id, ctx, cursor);
    check_layout(floats == def.floats, "a scalar block's float count");
    cursor += floats;
  }
  check_layout(cursor == out + input_floats(spec), "the row's total float count");
}

}  // namespace

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

void GameStateEncoder::encode_input(int player, const Rack& my_rack, bool apply_flip,
                                    float* out) const {
  assert(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], /*opp_rack=*/nullptr, apply_flip, out);
}

void GameStateEncoder::encode_input(int player, const Rack& my_rack, const Rack& opp_rack,
                                    bool apply_flip, float* out) const {
  assert(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], &opp_rack, apply_flip, out);
}

void GameStateEncoder::encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                                    bool apply_flip, float* out) const {
  assert(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp], score_diff,
             /*opp_rack=*/nullptr, apply_flip, out);
}

void GameStateEncoder::overwrite_score_diff(int score_diff, float* input_row) const {
  float* block =
    input_row + spatial_floats(spec_) + scalar_block_offset(spec_, ScalarBlockId::kScoreDiff);
  std::memset(block, 0, sizeof(float) * static_cast<size_t>(kScoreDiffThermoFloats));
  encode_score_diff_thermometer(score_diff, block);
}

}  // namespace scribblez
