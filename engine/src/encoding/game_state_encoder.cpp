#include "encoding/game_state_encoder.h"

#include "encoding/board_planes.h"
#include "encoding/input_encoder.h"
#include "game/glyph.h"
#include "game/tile.h"
#include "training/footprint_mask.h"
#include "util/assert.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace scribblez {

// TILE_COUNTS minus the tiles on `board` and, if non-null, in `held`.
static void tiles_off_board_and_hand(uint8_t out[27], const Board& board, const Rack* held) {
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
  if (held == nullptr) return;
  for (Tile t : held->tiles()) {
    if (t.is_empty()) continue;
    DEBUG_ASSERT(out[t] > 0);
    --out[t];
  }
}

void compute_unseen_pool(uint8_t out[27], const Board& board, const Rack& my_rack) {
  tiles_off_board_and_hand(out, board, &my_rack);
}

namespace {

// The tiles the POV player might play on their own next turn, for the
// self-reach plane: every unplayed tile except those the opponent is known to
// hold (`known_opp`, their open leave; null for a hidden-leaves spec). This
// includes the POV player's own rack, and the bag and the opponent's unknown
// tiles, which the POV player could still draw.
void compute_self_reach_pool(uint8_t out[27], const Board& board, const Rack* known_opp) {
  tiles_off_board_and_hand(out, board, known_opp);
}

// Everything the block writers read.
struct PovCtx {
  const Board& board;
  const Rack& my_rack;
  const Move& self_move;
  const Move& opp_move;
  int score_diff;
  const uint8_t* unseen;     // 27 per-tile counts, from compute_unseen_pool
  const uint8_t* self_pool;  // 27 per-tile counts, from compute_self_reach_pool
  const Rack* opp_leave;     // null iff the spec excludes the open-leaves block
};

// EXCHANGE and PASS (including the initial last-move placeholder) leave the
// plane all-zero.
int encode_placement_plane(const Move& m, float* out) {
  visit_placed_squares(m, [&](int r, int c) { out[r * kBoardSide + c] = 1.0f; });
  return 1;
}

// A square with no perpendicular neighbor has an all-ones mask, so it is
// written as all 26 letters legal. A 1 then means "legal here" on its own; an
// all-zero encoding would force the network to check the neighbors to read it.
void write_cross_check(const CrossCheck& cc, int cell, float* planes) {
  for (int l = 0; l < 26; ++l) {
    if (cc.mask & (1u << l)) planes[l * kBoardCells + cell] = 1.0f;
  }
}

// Horizontal A..Z, then vertical A..Z. A plane records only the per-square
// cross-check constraint; whether the main word itself is valid depends on
// the whole play and is not a per-square fact.
int encode_cross_check_planes(const Board& board, float* out) {
  float* h_planes = out;
  float* v_planes = out + kHorizontalCrossCheckPlanes * kBoardCells;
  // A horizontal word's cross words run down the columns, which is the
  // non-transposed cache; a vertical word's run along the rows.
  const auto& horizontal_play_cross = board.cross_checks(/*transposed=*/false);
  const auto& vertical_play_cross = board.cross_checks(/*transposed=*/true);

  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      if (!board.at(r, c).is_empty()) continue;
      const int out_ix = r * kBoardSide + c;
      write_cross_check(horizontal_play_cross[r * kBoardSide + c], out_ix, h_planes);
      write_cross_check(vertical_play_cross[c * kBoardSide + r], out_ix, v_planes);
    }
  }
  return kCrossCheckPlanes;
}

int encode_rack_counts(const Rack& my_rack, float* out) {
  for (Tile t : my_rack.tiles()) {
    if (!t.is_empty()) out[t.index()] += 1.0f;
  }
  return kRackCountFloats;
}

int encode_unseen_pool_thermometer(const uint8_t unseen[27], float* out) {
  int offset = 0;
  for (int i = 0; i < 27; ++i) {
    for (int j = 0; j < unseen[i]; ++j) out[offset + j] = 1.0f;
    offset += TILE_COUNTS[i];
  }
  DEBUG_ASSERT(offset == kUnseenPoolThermoFloats);
  return kUnseenPoolThermoFloats;
}

// TODO(score-diff resolution near the endgame): as the bag empties, the
// win/loss boundary in score difference becomes sharp; a two-point swing can
// flip the likely outcome, where mid-game it barely matters. A single linear
// scalar can express that, but the network must spend capacity learning the
// steep, phase-dependent transition exactly where endgame training positions
// are sparsest. To measure it, slice held-out win calibration by (tiles
// remaining, score difference) and look for win probability flattened across
// the flip point in the near-empty-bag, small-difference cells. If it shows,
// prefer a few nonlinear features (RBF bumps or bins, denser near 0) over a
// full thermometer, and change the move set model's score feature
// (move_set_encoder.cpp) to match. The decisive endgame itself belongs to the
// endgame solver (docs/roadmap.md item 7), not to finer value-net input.
int encode_score_diff_scalar(int score_diff, float* out) {
  out[0] = float(score_diff) / kScoreDiffInputScale;
  return kScoreDiffInputFloats;
}

int encode_move_meta(const Move& self_move, const Move& opp_move, float* out) {
  out[int(self_move.type())] = 1.0f;
  out[kMoveMetaTypeFloats] = float(self_move.num_glyphs());
  float* opp = out + kMoveMetaFloatsPerMove;
  opp[int(opp_move.type())] = 1.0f;
  opp[kMoveMetaTypeFloats] = float(opp_move.num_glyphs());
  return kMoveMetaFloats;
}

// The cells some this-turn move drawing on `pool` could cover.
int encode_reach_plane(const Board& board, const uint8_t* pool, float* out) {
  footprint_reachable_cells(board, pool, kMaskTileBudget, out);
  return 1;
}

// Writes one block at `out` and returns its plane count.
int encode_spatial_block(SpatialBlockId id, const PovCtx& ctx, float* out) {
  switch (id) {
    case SpatialBlockId::kBoard:
      static_assert(BoardPlanes::kPlanes == kBoardBlockPlanes,
                    "the registry's board block must match the shared board-plane encoder");
      BoardPlanes::encode(ctx.board, out);
      return BoardPlanes::kPlanes;
    case SpatialBlockId::kSelfPlacement:
      return encode_placement_plane(ctx.self_move, out);
    case SpatialBlockId::kOppPlacement:
      return encode_placement_plane(ctx.opp_move, out);
    case SpatialBlockId::kCrossChecks:
      return encode_cross_check_planes(ctx.board, out);
    case SpatialBlockId::kOppReach:
      return encode_reach_plane(ctx.board, ctx.unseen, out);
    case SpatialBlockId::kSelfReach:
      return encode_reach_plane(ctx.board, ctx.self_pool, out);
  }
  std::abort();  // unreachable: the switch covers every SpatialBlockId
}

// Writes one block at `out` and returns its float count.
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
    case ScalarBlockId::kOppLeaveCounts:
      return encode_rack_counts(*ctx.opp_leave, out);
  }
  std::abort();  // unreachable: the switch covers every ScalarBlockId
}

// Enabled in release builds too: it costs a few integer compares per row, and
// a block writer disagreeing with the registry would corrupt memory.
void check_layout(bool ok, const char* what) {
  if (ok) return;
  std::fprintf(stderr, "input encode: %s disagrees with the layout registry\n", what);
  std::abort();
}

// The shared encoder behind every GameStateEncoder::encode_input* method.
// `score_diff` is the POV player's lead.
void encode_pov(const InputEncodingSpec& spec, const Board& board, const Rack& my_rack,
                const Move& self_move, const Move& opp_move, int score_diff, const Rack* opp_leave,
                float* out) {
  DEBUG_ASSERT(self_move.transposed() == board.transposed());
  DEBUG_ASSERT(opp_move.transposed() == board.transposed());
  check_layout(!spec.opp_leave_input || opp_leave != nullptr,
               "an open-leaves spec encoded without the opponent leave");
  std::memset(out, 0, sizeof(float) * size_t(input_floats(spec)));
  // The cross-check and reach planes read the board's move-generation caches.
  // A no-op if they are already built.
  board.ensure_movegen_caches(*spec.dict);
  const Rack* known_opp = spec.opp_leave_input ? opp_leave : nullptr;
  uint8_t unseen[27];
  compute_unseen_pool(unseen, board, my_rack);
  uint8_t self_pool[27];
  compute_self_reach_pool(self_pool, board, known_opp);
  const PovCtx ctx{board, my_rack, self_move, opp_move, score_diff, unseen, self_pool, known_opp};

  float* cursor = out;
  for (const SpatialBlockDef& def : kSpatialBlocks) {
    const int planes = encode_spatial_block(def.id, ctx, cursor);
    check_layout(planes == def.planes, "a spatial block's plane count");
    cursor += planes * kBoardCells;
  }
  check_layout(cursor == out + spatial_floats(), "the spatial section's total");
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
  last_move_by_[active_] = move;
  active_ = 1 - active_;
  ++turn_index_;
}

GameStateEncoder GameStateEncoder::transpose() const {
  GameStateEncoder out = *this;
  out.board_ = board_.transpose();
  for (Move& m : out.last_move_by_) m = m.transpose();
  return out;
}

void GameStateEncoder::encode_input(int player, const Rack& my_rack, float* out) const {
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], /*opp_leave=*/nullptr, out);
}

void GameStateEncoder::encode_input(int player, const Rack& my_rack, const Rack& opp_leave,
                                    float* out) const {
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp],
             scores_[player] - scores_[opp], &opp_leave, out);
}

void GameStateEncoder::encode_input_with_score_diff(int player, const Rack& my_rack, int score_diff,
                                                    float* out) const {
  RELEASE_ASSERT(player == 0 || player == 1);
  const int opp = 1 - player;
  encode_pov(spec_, board_, my_rack, last_move_by_[player], last_move_by_[opp], score_diff,
             /*opp_leave=*/nullptr, out);
}

void GameStateEncoder::overwrite_score_diff(int score_diff, float* input_row) const {
  float* block =
    input_row + spatial_floats() + scalar_block_offset(spec_, ScalarBlockId::kScoreDiff);
  encode_score_diff_scalar(score_diff, block);
}

void encode_post_move_row(const GameStateEncoder& pre, int mover, const Rack& my_rack,
                          const Move& mv, const Rack& opp_leave, float* out) {
  Rack leave = my_rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  GameStateEncoder post = pre;
  post.apply_move(mv);
  post.encode_input(mover, leave, opp_leave, out);
}

}  // namespace scribblez
