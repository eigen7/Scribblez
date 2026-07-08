#pragma once

// Input-encoding configuration and row layout for the value models' input
// tensor. The encoding itself is performed by GameStateEncoder (see
// encoding/game_state_encoder.h); this header owns the layout: an
// InputEncodingSpec picks the blocks a run encodes, and the block registry
// below (the input-side analog of training_targets.h's AllTargets) is the
// single source of truth for their order and sizes. The encoder writes the
// blocks by walking the registry, the FFI advertises shapes computed from it,
// and every consumer that needs a block's location asks it -- so an offset
// cannot drift from the write order.
//
// Layout: `spatial_planes(spec)` 15x15 planes (channel-major, PyTorch
// (C, H, W) via a zero-copy reshape), then `scalar_floats(spec)` scalars.
// Block contents:
//
//   Spatial blocks
//     kBoard          31 planes: letter A..Z presence (including designated
//                     blanks), the blank marker, and the DLS/TLS/DWS/TWS
//                     premium masks (the trunk's only absolute-position
//                     anchor). See board_planes.h.
//     kSelfPlacement   1 plane: squares the POV player covered in their most
//                     recent turn.
//     kOppPlacement    1 plane: the same for the opponent. Together the two
//                     placement planes encode the last two plies.
//     kCrossChecks    52 planes: horizontal then vertical cross-check letters
//                     A..Z -- plane L marks empty squares where placing L
//                     fuses with a neighbor along that axis and satisfies the
//                     lexicon's cross-check mask.
//     kContingent      3 planes (contingent runs only): the contingent-draw
//                     potential maps -- best / draw-weighted / rack-alone
//                     next-turn plays painted onto their placed cells (see
//                     contingent_map.h).
//
//   Scalar blocks (POV-visible information only; values normalized, most to
//   [0, 1] -- kScoreDiff is the one signed exception)
//     kRackCounts     27: the POV rack's raw per-tile counts (A..Z, blank).
//     kUnseenPool    100: unseen-pool thermometer, one slot per physical tile
//                     grouped per letter (TILE_COUNTS regions).
//     kScoreDiff       1: (score_active - score_opp) / kScoreDiffInputScale, a
//                     signed scalar (not clipped).
//     kMoveMeta        8: last-2-move metadata (self then opponent): a 3-way
//                     PLAY/EXCHANGE/PASS one-hot + num_glyphs each.
//     kContingent     56 (contingent runs only): per drawable tile kind the
//                     best contingent score over all lanes (27), the same
//                     draw-weighted (27), the expected best, the rack-alone
//                     best.
//     kOppLeaveCounts 27 (open-leaves runs only): per-tile counts of the
//                     opponent's retained leave -- the tiles they kept from
//                     their last move, excluding their hidden replenishment
//                     draws. The one exception to the POV-visibility rule,
//                     available only under the open-leaves information
//                     condition (spec.opp_leave_input). All zeros when the
//                     opponent has not acted or their last move kept nothing
//                     (a bingo).
//
// The conditional blocks sit at the TAILS of their sections, so a smaller
// arm's row is a larger arm's row with the tails spliced out; consumers that
// serve multiple arms (e.g. the dashboard running a base model on full-layout
// rows) rely on that prefix property.
//
// Symmetry: the board is invariant under the diagonal flip (r,c) -> (c,r);
// `apply_flip` transposes every spatial plane, and all scalars are
// flip-invariant.

namespace scribblez {

class Dictionary;

// Per-run input-encoding configuration: the lexicon the lexical features
// derive from, whether the contingent-draw potential blocks are computed and
// included, and whether the opponent's retained leave is included
// ("open-leaves" -- an experiment-only information condition in which the
// tiles a player KEPT from their last move are public while their
// replenishment draws stay hidden; the leave is the Bayesian-inferable part
// of a rack, so this hands the model a perfect belief posterior; see
// docs/sim_residual_feedback.md). Chosen once per process (baked into the
// FFI session) and carried by every encoder; a model's contingent arm is
// recorded in its ONNX metadata_props ("contingent_features") so serving
// consumers can recover it (open-leaves models are research instruments and
// are not exported for serving). The row layout a spec selects is owned by
// the block registry below.
struct InputEncodingSpec {
  const Dictionary* dict;
  bool contingent_features;
  bool opp_leave_input = false;
};

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
// Score differential as a single signed scalar: (score_active - score_opp)
// divided by kScoreDiffInputScale, not clipped. The move set evaluation model's
// resultant-diff move feature shares this representation, so a post-move
// differential is the pre-move differential plus the move score, a plain sum.
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

// Whether `spec` includes a scalar block -- the single predicate every walk
// over kScalarBlocks (sizing, offsets, and the encoder itself) shares.
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
