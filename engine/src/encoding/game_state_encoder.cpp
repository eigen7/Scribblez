#include "encoding/game_state_encoder.h"

#include "encoding/board_planes.h"
#include "encoding/contingent_map.h"
#include "encoding/input_encoder.h"
#include "game/glyph.h"
#include "game/tile.h"
#include "util/assert.h"
#include "util/math.h"

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
  for (int i = 0; i < 27; ++i) out[i] = uint8_t(TILE_COUNTS[i]);
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      Tile t = g.rack_tile();
      DEBUG_ASSERT(out[t] > 0);
      --out[t];
    }
  }
  for (Tile t : my_rack.tiles()) {
    if (t.is_empty()) continue;
    DEBUG_ASSERT(out[t] > 0);
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
  const Rack* opp_leave;            // null iff the spec excludes the open-leaves block
};

// One plane at `out`, marking the squares `m` placed tiles on. EXCHANGE, PASS,
// and game-start place nothing and leave it all-zero.
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

// Set `cc`'s legal letters at `cell` across a 26-plane block. A square with no
// perpendicular run constrains nothing, and says so by staying zero rather
// than by setting all 26 of its planes.
void write_cross_check(const CrossCheck& cc, int cell, float* planes) {
  if (!cc.has_neighbor) return;
  for (int l = 0; l < 26; ++l) {
    if (cc.mask & (1u << l)) planes[l * kBoardCells + cell] = 1.0f;
  }
}

// The kCrossCheckPlanes cross-check planes at `out`: the letters a HORIZONTAL
// word may place on each empty square (A..Z), then the same for a VERTICAL
// word.
//
// A word running along one axis places exactly one tile in each lane of the
// other, so the letters it may put on an empty square are exactly that
// square's PERPENDICULAR cross-check set -- the letters completing a valid
// word with the tiles the square abuts across the word's direction. That holds
// at every word length, which is what makes it a per-square fact worth a
// plane. The word's own validity is not such a fact: it depends on the whole
// play, so intersecting it in here would deny letters that a longer word
// legitimately places (an `A` on a square reading `_XI` is illegal alone and
// legal in AXIOM) while collapsing both blocks onto one-tile legality.
int encode_cross_check_planes(const Board& board, bool flip, float* out) {
  float* h_planes = out;
  float* v_planes = out + kHorizontalCrossCheckPlanes * kBoardCells;
  // A horizontal word's cross words run down the columns, which is what the
  // non-transposed cache holds; a vertical word's run along the rows.
  const auto& horizontal_play_cross = board.cross_checks(/*transposed=*/false);
  const auto& vertical_play_cross = board.cross_checks(/*transposed=*/true);

  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      if (!board.at(r, c).is_empty()) continue;
      const int out_ix = plane_idx(r, c, flip);
      write_cross_check(horizontal_play_cross[r * kBoardSide + c], out_ix, h_planes);
      write_cross_check(vertical_play_cross[c * kBoardSide + r], out_ix, v_planes);
    }
  }
  return kCrossCheckPlanes;
}

// The active player's rack as raw per-tile counts.
int encode_rack_counts(const Rack& my_rack, float* out) {
  for (Tile t : my_rack.tiles()) {
    if (!t.is_empty()) out[t.index()] += 1.0f;
  }
  return kRackCountFloats;
}

// A per-letter thermometer: letter i owns a region of width TILE_COUNTS[i]
// whose first unseen[i] slots are 1.0, the holes falling at the tail. Regions
// are concatenated in tile order.
int encode_unseen_pool_thermometer(const uint8_t unseen[27], float* out) {
  int offset = 0;
  for (int i = 0; i < 27; ++i) {
    for (int j = 0; j < unseen[i]; ++j) out[offset + j] = 1.0f;
    offset += TILE_COUNTS[i];
  }
  DEBUG_ASSERT(offset == kUnseenPoolThermoFloats);
  return kUnseenPoolThermoFloats;
}

// Score differential as a single signed scalar (kScoreDiffInputFloats floats):
// (score_active - score_opp) / kScoreDiffInputScale, not clipped.
//
// ANALYSIS TODO (score-diff resolution near the endgame): as the bag empties
// the win/draw/loss boundary in score differential becomes sharp and
// phase-conditional -- a two-point swing (e.g. +15 vs +17) can flip the likely
// outcome, whereas mid-game it is nearly flat. A single scalar can express this
// (the first projection can amplify small differences), but its smoothness bias
// makes a steep, phase-gated transition something the trunk must spend capacity
// to learn, exactly where true-endgame training positions are sparsest. Measure
// it: slice held-out WLD calibration/Brier by (tiles-remaining, score-diff) and
// look at the near-empty-bag, small-|diff| cells for under-sharpening (win prob
// flattened across the flip point). If it shows, prefer a compact nonlinear
// featurization (a handful of RBF/bins, denser near 0) over widening back to a
// full thermometer, and apply the same basis to the move set evaluation
// model's resultant-diff move feature (move_set_encoder) so the two stay on
// one representation. The principled answer for the decisive endgame is the
// negamax solver (docs/roadmap.md, D3), not finer value-net input resolution.
int encode_score_diff_scalar(int score_diff, float* out) {
  out[0] = float(score_diff) / kScoreDiffInputScale;
  return kScoreDiffInputFloats;
}

// Last-2-move metadata (kMoveMetaFloats floats): for the POV player's and then
// the opponent's most recent move, a move-type one-hot (indexed by MoveType)
// followed by num_glyphs.
int encode_move_meta(const Move& self_move, const Move& opp_move, float* out) {
  out[int(self_move.type())] = 1.0f;
  out[kMoveMetaTypeFloats] = float(self_move.num_glyphs());
  float* opp = out + kMoveMetaFloatsPerMove;
  opp[int(opp_move.type())] = 1.0f;
  opp[kMoveMetaTypeFloats] = float(opp_move.num_glyphs());
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
      return encode_score_diff_scalar(ctx.score_diff, out);
    case ScalarBlockId::kMoveMeta:
      return encode_move_meta(ctx.self_move, ctx.opp_move, out);
    case ScalarBlockId::kContingent:
      ctx.contingent->encode_scalars(out);
      return kContingentScalarFloats;
    case ScalarBlockId::kOppLeaveCounts:
      return encode_rack_counts(*ctx.opp_leave, out);
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
                const Move& self_move, const Move& opp_move, int score_diff, const Rack* opp_leave,
                bool apply_flip, float* out) {
  check_layout(!spec.opp_leave_input || opp_leave != nullptr,
               "an open-leaves spec encoded without the opponent leave");
  std::memset(out, 0, sizeof(float) * size_t(input_floats(spec)));
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
                   spec.opp_leave_input ? opp_leave : nullptr};

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
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], /*opp_leave=*/nullptr, apply_flip, out);
}

void GameStateEncoder::encode_input(int player, const Rack& my_rack, const Rack& opp_leave,
                                    bool apply_flip, float* out) const {
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], &opp_leave, apply_flip, out);
}

void GameStateEncoder::encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                                    bool apply_flip, float* out) const {
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp], score_diff,
             /*opp_leave=*/nullptr, apply_flip, out);
}

void GameStateEncoder::overwrite_score_diff(int score_diff, float* input_row) const {
  float* block =
    input_row + spatial_floats(spec_) + scalar_block_offset(spec_, ScalarBlockId::kScoreDiff);
  encode_score_diff_scalar(score_diff, block);
}

void encode_post_move_row(const GameStateEncoder& pre, int mover, const Rack& my_rack,
                          const Move& mv, const Rack& opp_leave, float* out) {
  Rack leave = my_rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  GameStateEncoder post = pre;
  post.apply_move(mv);
  if (pre.spec().opp_leave_input) {
    post.encode_input(mover, leave, opp_leave, /*apply_flip=*/false, out);
  } else {
    post.encode_input(mover, leave, /*apply_flip=*/false, out);
  }
}

}  // namespace scribblez
