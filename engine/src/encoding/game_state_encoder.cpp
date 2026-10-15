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

// Board::unseen_tiles(), with a null `held` holding nothing, as counts.
static void tiles_off_board_and_hand(uint8_t out[TILE_KINDS], const Board& board,
                                     const Rack* held) {
  const TileCounts unseen = board.unseen_tiles(held == nullptr ? Rack() : *held);
  for (int i = 0; i < TILE_KINDS; ++i) out[i] = uint8_t(unseen.count(Tile::of(i)));
}

void compute_unseen_pool(uint8_t out[TILE_KINDS], const Board& board, const Rack& my_rack) {
  tiles_off_board_and_hand(out, board, &my_rack);
}

namespace {

// The tiles the POV player might play on their own next turn, for the
// self-reach plane: every unplayed tile except those the opponent is known to
// hold (`known_opp`, their open leave; null for a hidden-leaves spec). This
// includes the POV player's own rack, and the bag and the opponent's unknown
// tiles, which the POV player could still draw.
void compute_self_reach_pool(uint8_t out[TILE_KINDS], const Board& board, const Rack* known_opp) {
  tiles_off_board_and_hand(out, board, known_opp);
}

// Everything the block writers read.
struct PovCtx {
  const Board& board;
  const Rack& my_rack;
  const Move& self_move;
  const Move& opp_move;
  int score_diff;
  const uint8_t* unseen;     // TILE_KINDS per-tile counts, from compute_unseen_pool
  const uint8_t* self_pool;  // TILE_KINDS per-tile counts, from compute_self_reach_pool
  const Rack* opp_leave;     // null iff the spec excludes the open-leaves block
};

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
  uint8_t unseen[TILE_KINDS];
  compute_unseen_pool(unseen, board, my_rack);
  uint8_t self_pool[TILE_KINDS];
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

void encode_post_move_row(const GameStateEncoder& pre, int mover, const Rack& my_rack,
                          const Move& mv, const Rack& opp_leave, float* out) {
  Rack leave = my_rack;
  for (int i = 0; i < mv.num_glyphs(); ++i) leave.remove(mv.glyph(i).rack_tile());
  GameStateEncoder post = pre;
  post.apply_move(mv);
  post.encode_input(mover, leave, opp_leave, out);
}

}  // namespace scribblez
