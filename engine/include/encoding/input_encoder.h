#pragma once

// The layout of the position evaluation model's input row. GameStateEncoder
// writes the row; this header defines it. The block registry below is the one
// source of truth for block order and sizes: the encoder writes by walking it,
// the FFI computes shapes from it, and consumers look up block offsets in it,
// so an offset cannot drift from the write order.
//
// A row is spatial_planes() 15x15 planes, channel-major (so PyTorch reshapes it
// to (C, H, W) without a copy), followed by scalar_floats(spec) scalars. All
// features are from the POV player's side of the table, and all are visible to
// that player except kOppLeaveCounts.
//
//   Spatial blocks
//     kBoard          letter planes A..Z (a designated blank sets its letter),
//                     a blank-marker plane, and DLS/TLS/DWS/TWS premium planes.
//                     The premiums are the network's only fix on absolute
//                     position. See board_planes.h.
//     kSelfPlacement  the squares the POV player covered on their last turn.
//     kOppPlacement   the same for the opponent.
//     kCrossChecks    26 horizontal planes, then 26 vertical: plane L marks the
//                     empty squares where L satisfies the cross-check for a
//                     word along that axis. A square with no perpendicular
//                     neighbor allows every letter, so a 1 always means "L is
//                     legal here" without reference to the neighbors.
//     kOppReach       the squares the opponent, who moves next, could cover
//                     with tiles they might hold (the unseen pool): cells
//                     covered by some footprint that touches the existing
//                     tiles, passes the cross-checks, and has its letters
//                     available. Precomputed because a conv trunk derives this
//                     poorly. See training/footprint_mask.h.
//     kSelfReach      the same for the POV player's own next turn, assuming the
//                     opponent passes, over every unplayed tile the opponent is
//                     not known to hold. The POV player's actual next move was
//                     often already legal before the opponent's reply, so this
//                     is a good starting point for the self-placement heads.
//
//   Scalar blocks
//     kRackCounts     the POV rack's per-tile counts (A..Z, blank), unscaled.
//     kUnseenPool     a thermometer with one slot per physical tile, in
//                     per-tile regions of width TILE_COUNTS[t]: region t has
//                     its first unseen[t] slots set.
//     kScoreDiff      (score_pov - score_opp) / kScoreDiffInputScale, unclipped.
//     kMoveMeta       for the POV player's last move, then the opponent's: a
//                     one-hot over MoveType (PLAY, EXCHANGE, PASS), then the
//                     number of tiles placed or exchanged, unscaled.
//     kOppLeaveCounts open leaves only: per-tile counts of what the opponent
//                     kept from their last move. All zeros if they have not
//                     moved or kept nothing.
//
// The one conditional block, kOppLeaveCounts, ends the row, so a hidden-leaves
// row is a prefix of the open-leaves row for the same position. Consumers that
// serve several arms, e.g. a dashboard running a hidden-leaves model on
// open-leaves rows, rely on this. Keep any new conditional block at the end of
// its section.
//
// Training augments with the diagonal transpose (r,c) -> (c,r), under which
// the board is symmetric. The encoder knows nothing of it: an augmented row is
// the encoding of a transposed position (Board::transpose). Its spatial planes
// come out transposed, with the two kCrossChecks halves swapped since that is
// the one block whose contents name an axis, and its scalars are unchanged.

namespace scribblez {

class Dictionary;

// Which input arm to encode, carried by every encoder. A model's arm is
// recorded in its ONNX metadata so consumers can recover it.
//
// opp_leave_input selects the open-leaves arm, for the face-up-leaves variant
// the project develops in (docs/roadmap.md): the tiles a player kept from their
// last move are public, and only their draws stay hidden.
struct InputEncodingSpec {
  const Dictionary* dict;
  bool opp_leave_input = false;
};

// The version of the encoding's semantics. Bump it whenever a change alters
// what a position encodes to without changing any block's width. A model is
// only valid with the encoding it was trained on, and a semantic change leaves
// every shape intact, so nothing else would catch the mismatch. Exporters
// stamp the version into ONNX metadata, and the engine's loader rejects a model
// with a different one. A model without the entry reads as version 0.
//
//   1: cross-check plane bug fixes
//   2: a square with no perpendicular neighbor sets all 26 cross-check planes
inline constexpr int kInputEncodingVersion = 2;

inline constexpr int kBoardSide = 15;
inline constexpr int kBoardCells = kBoardSide * kBoardSide;  // 225

inline constexpr int kBoardBlockPlanes = 31;  // == BoardPlanes::kPlanes, asserted by the writer
inline constexpr int kHorizontalCrossCheckPlanes = 26;
inline constexpr int kVerticalCrossCheckPlanes = 26;
inline constexpr int kCrossCheckPlanes = kHorizontalCrossCheckPlanes + kVerticalCrossCheckPlanes;
inline constexpr int kRackCountFloats = 27;
inline constexpr int kUnseenPoolThermoFloats = 100;  // == sum(TILE_COUNTS) for English Scrabble
// The move set evaluation model's post-move score-difference feature uses the
// same scale, so it is simply this input plus the move's scaled score.
inline constexpr int kScoreDiffInputFloats = 1;
inline constexpr float kScoreDiffInputScale = 100.0f;
inline constexpr int kMoveMetaTypeFloats = 3;
inline constexpr int kMoveMetaFloatsPerMove = kMoveMetaTypeFloats + 1;
inline constexpr int kMoveMetaFloats = 2 * kMoveMetaFloatsPerMove;
inline constexpr int kOppLeaveCountFloats = 27;

// ---- Block registry ---------------------------------------------------------

enum class SpatialBlockId {
  kBoard,
  kSelfPlacement,
  kOppPlacement,
  kCrossChecks,
  kOppReach,
  kSelfReach
};
enum class ScalarBlockId { kRackCounts, kUnseenPool, kScoreDiff, kMoveMeta, kOppLeaveCounts };

struct SpatialBlockDef {
  SpatialBlockId id;
  int planes;
};
struct ScalarBlockDef {
  ScalarBlockId id;
  int floats;
  bool opp_leave_only = false;  // included iff spec.opp_leave_input
};

// The row's blocks, in encode order.
inline constexpr SpatialBlockDef kSpatialBlocks[] = {
  {SpatialBlockId::kBoard, kBoardBlockPlanes},
  {SpatialBlockId::kSelfPlacement, 1},
  {SpatialBlockId::kOppPlacement, 1},
  {SpatialBlockId::kCrossChecks, kCrossCheckPlanes},
  {SpatialBlockId::kOppReach, 1},
  {SpatialBlockId::kSelfReach, 1},
};
inline constexpr ScalarBlockDef kScalarBlocks[] = {
  {ScalarBlockId::kRackCounts, kRackCountFloats},
  {ScalarBlockId::kUnseenPool, kUnseenPoolThermoFloats},
  {ScalarBlockId::kScoreDiff, kScoreDiffInputFloats},
  {ScalarBlockId::kMoveMeta, kMoveMetaFloats},
  {ScalarBlockId::kOppLeaveCounts, kOppLeaveCountFloats, true},
};

// Every walk over kScalarBlocks (sizing, offsets, the encoder) must use this.
inline bool scalar_block_included(const ScalarBlockDef& def, const InputEncodingSpec& spec) {
  return !def.opp_leave_only || spec.opp_leave_input;
}

// ---- Layout queries ---------------------------------------------------------
//
// The spatial section has no conditional block, so only the scalar queries
// take the spec.

inline constexpr int spatial_planes() {  // 87
  int planes = 0;
  for (const SpatialBlockDef& def : kSpatialBlocks) planes += def.planes;
  return planes;
}
inline constexpr int spatial_floats() { return spatial_planes() * kBoardCells; }
int spatial_block_plane0(SpatialBlockId id);

int scalar_floats(const InputEncodingSpec& spec);  // 136; 163 under open leaves
int input_floats(const InputEncodingSpec& spec);

// A block's offset within the scalar section. Aborts if the spec excludes the
// block.
int scalar_block_offset(const InputEncodingSpec& spec, ScalarBlockId id);

}  // namespace scribblez
