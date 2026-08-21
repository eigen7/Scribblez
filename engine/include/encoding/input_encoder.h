#pragma once

// Row layout for the value models' input tensor. GameStateEncoder does the
// encoding; this header owns the layout. An InputEncodingSpec picks the blocks
// a run encodes, and the block registry below is the single source of truth for
// their order and sizes -- the encoder writes by walking it, the FFI advertises
// shapes computed from it, and every consumer asks it for a block's location,
// so an offset cannot drift from the write order.
//
// A row is `spatial_planes(spec)` 15x15 planes (channel-major, PyTorch
// (C, H, W) via a zero-copy reshape) followed by `scalar_floats(spec)` scalars.
//
//   Spatial blocks
//     kBoard          letter A..Z presence (designated blanks included), the
//                     blank marker, and the DLS/TLS/DWS/TWS premium masks --
//                     the trunk's only absolute-position anchor. See
//                     board_planes.h.
//     kSelfPlacement  squares the POV player covered on their most recent turn.
//     kOppPlacement   the same for the opponent, so the two together encode the
//                     last two plies.
//     kCrossChecks    horizontal then vertical: plane L marks empty squares
//                     where placing L satisfies the lexicon's cross-check mask
//                     along that axis.
//     kContingent     the contingent-draw potential maps -- best /
//                     draw-weighted / rack-alone next-turn plays painted onto
//                     their placed cells (see contingent_map.h).
//
//   Scalar blocks (POV-visible information only, normalized to [0, 1] but for
//   the signed kScoreDiff)
//     kRackCounts     the POV rack's raw per-tile counts (A..Z, blank).
//     kUnseenPool     thermometer, one slot per physical tile, grouped per
//                     letter (TILE_COUNTS regions).
//     kScoreDiff      (score_active - score_opp) / kScoreDiffInputScale, signed
//                     and unclipped.
//     kMoveMeta       per last move, self then opponent: a 3-way
//                     PLAY/EXCHANGE/PASS one-hot plus num_glyphs.
//     kContingent     per drawable tile kind the best contingent score over all
//                     lanes, the same draw-weighted, then the expected best and
//                     the rack-alone best.
//     kOppLeaveCounts per-tile counts of the tiles the opponent kept from their
//                     last move, their hidden replenishment draws excluded. The
//                     one exception to POV visibility, and all zeros when the
//                     opponent has not acted or bingoed.
//
// The conditional blocks sit at the TAILS of their sections, so a smaller arm's
// row is a larger arm's row with the tails spliced out. Consumers serving
// several arms (e.g. a dashboard running a base model on full-layout rows) rely
// on that prefix property.
//
// The board is invariant under the diagonal flip (r,c) -> (c,r), so
// `apply_flip` transposes every spatial plane and leaves the scalars alone.
// The flip also exchanges the axes, so the kCrossChecks halves swap as they
// transpose -- the only block whose contents name an axis.

namespace scribblez {

class Dictionary;

// Per-run input-encoding configuration, chosen once per process (baked into
// the FFI session) and carried by every encoder. A model's contingent arm is
// recorded in its ONNX metadata_props so serving consumers can recover it.
//
// "Open leaves" is an experiment-only information condition in which the tiles
// a player KEPT from their last move are public while their replenishment draws
// stay hidden. The leave is the Bayesian-inferable part of a rack, so this
// hands the model a perfect belief posterior (see
// docs/sim_residual_feedback.md); such models are research instruments and are
// never exported for serving.
struct InputEncodingSpec {
  const Dictionary* dict;
  bool contingent_features;
  bool opp_leave_input = false;
};

// The board-row encoding SEMANTICS version: bumped whenever an encoder change
// alters what the same position encodes to without changing any block width. A
// checkpoint is only valid with the encoding its training rows used, and
// nothing structural detects a semantic drift -- the arm booleans and every
// plane count survive it -- so the version rides the exported ONNX metadata,
// where the engine-side loader rejects a stale model instead of silently
// feeding it off-distribution rows. Absent in exports predating the entry,
// which read as version 0 -- the version at the entry's introduction.
//
//   1: cross-check planes bug fixes
inline constexpr int kInputEncodingVersion = 1;

inline constexpr int kBoardSide = 15;
inline constexpr int kBoardCells = kBoardSide * kBoardSide;  // 225

// Block-content sizes (referenced by the registry and the block writers).
inline constexpr int kBoardBlockPlanes = 31;  // == BoardPlanes::kPlanes (asserted at the writer)
inline constexpr int kHorizontalCrossCheckPlanes = 26;
inline constexpr int kVerticalCrossCheckPlanes = 26;
inline constexpr int kCrossCheckPlanes = kHorizontalCrossCheckPlanes + kVerticalCrossCheckPlanes;
inline constexpr int kContingentPlanes = 3;  // max / draw-weighted / rack-alone potential
inline constexpr int kRackCountFloats = 27;
inline constexpr int kUnseenPoolThermoFloats = 100;  // == sum(TILE_COUNTS) for English Scrabble
// The move set evaluation model's resultant-diff move feature shares this
// representation, so a post-move differential is the pre-move differential plus
// the move score, a plain sum.
inline constexpr int kScoreDiffInputFloats = 1;
inline constexpr float kScoreDiffInputScale = 100.0f;
inline constexpr int kMoveMetaTypeFloats = 3;  // PLAY / EXCHANGE / PASS one-hot
inline constexpr int kMoveMetaFloatsPerMove = kMoveMetaTypeFloats + 1;  // + num_glyphs
inline constexpr int kMoveMetaFloats = 2 * kMoveMetaFloatsPerMove;      // self + opp = 8
// Per-kind best + per-kind draw-weighted best, then expected best + rack-alone best.
inline constexpr int kContingentScalarFloats = 27 + 27 + 2;  // 56
inline constexpr int kOppLeaveCountFloats = 27;              // open-leaves arm only

// ---- Block registry ---------------------------------------------------------

enum class SpatialBlockId { kBoard, kSelfPlacement, kOppPlacement, kCrossChecks, kContingent };
enum class ScalarBlockId {
  kRackCounts,
  kUnseenPool,
  kScoreDiff,
  kMoveMeta,
  kContingent,
  kOppLeaveCounts
};

struct SpatialBlockDef {
  SpatialBlockId id;
  int planes;
  bool contingent_only;  // included iff spec.contingent_features
};
struct ScalarBlockDef {
  ScalarBlockId id;
  int floats;
  bool contingent_only;
  bool opp_leave_only = false;  // included iff spec.opp_leave_input
};

// The row's blocks in encode order. GameStateEncoder writes by walking these
// tables, so a block's offset is definitionally where the walk puts it.
inline constexpr SpatialBlockDef kSpatialBlocks[] = {
  {SpatialBlockId::kBoard, kBoardBlockPlanes, false},
  {SpatialBlockId::kSelfPlacement, 1, false},
  {SpatialBlockId::kOppPlacement, 1, false},
  {SpatialBlockId::kCrossChecks, kCrossCheckPlanes, false},
  {SpatialBlockId::kContingent, kContingentPlanes, true},
};
inline constexpr ScalarBlockDef kScalarBlocks[] = {
  {ScalarBlockId::kRackCounts, kRackCountFloats, false},
  {ScalarBlockId::kUnseenPool, kUnseenPoolThermoFloats, false},
  {ScalarBlockId::kScoreDiff, kScoreDiffInputFloats, false},
  {ScalarBlockId::kMoveMeta, kMoveMetaFloats, false},
  {ScalarBlockId::kContingent, kContingentScalarFloats, true},
  {ScalarBlockId::kOppLeaveCounts, kOppLeaveCountFloats, false, true},
};

// The single predicate every walk over kScalarBlocks shares -- sizing, offsets,
// and the encoder itself.
inline bool scalar_block_included(const ScalarBlockDef& def, const InputEncodingSpec& spec) {
  return (!def.contingent_only || spec.contingent_features) &&
         (!def.opp_leave_only || spec.opp_leave_input);
}

// ---- Layout queries (walk the registry under `spec`) ------------------------

int spatial_planes(const InputEncodingSpec& spec);  // 88 full / 85 base
int scalar_floats(const InputEncodingSpec& spec);   // 992 full / 936 base; +27 open-leaves
int spatial_floats(const InputEncodingSpec& spec);  // spatial_planes * kBoardCells
int input_floats(const InputEncodingSpec& spec);    // spatial + scalar

// First plane / first scalar offset of a block under `spec`. The block must be
// included by the spec (asking for an excluded block aborts).
int spatial_block_plane0(const InputEncodingSpec& spec, SpatialBlockId id);
int scalar_block_offset(const InputEncodingSpec& spec, ScalarBlockId id);

}  // namespace scribblez
