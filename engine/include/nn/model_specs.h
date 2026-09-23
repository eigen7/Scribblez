#pragma once

#include "encoding/input_encoder.h"
#include "nn/onnx_metadata.h"
#include "training/move_set_encoder.h"
#include "training/training_targets.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

// Each served model family, declared as data: its graph identity, the sizing
// of its dynamic axis, the encoding versions it must match, and its input and
// output tensors as type lists of descriptors. This is the one file where
// family specifics live. NeuralNet<Spec> turns a spec into a TensorRT runtime
// with no family-specific code, so adding a family means writing a spec here.
//
// A family served through TrtEvalService also needs Batch staging overloads in
// trt_eval_service.cpp. The move-proposal specs are not: their handoff outputs
// do not fit the service's per-row decode, so agent/move_proposal_nets.h drives
// them through NeuralNet directly.
//
// Carries no CUDA or TensorRT dependency, so GPU-free code (agent interfaces,
// test stubs) can include it.

namespace scribblez {
namespace nn {

// ---------- tensor descriptors ------------------------------------------

// One engine I/O tensor. A descriptor declares:
//   - kName: the ONNX tensor name, the contract string shared with the Python
//     model and exporter
//   - Elem: the element type of its host buffer
//   - kRowElems: the row width the staging and decode code assumes, or 0 when
//     the model decides it (e.g. board widths, which depend on the encoding arm)
//   - kDynamic: whether the tensor rides the spec's dynamic row axis
// The loader validates all of it against the model at load time.
template <typename T>
concept TensorDescriptor = requires {
  { T::kName } -> std::convertible_to<const char*>;
  typename T::Elem;
  { T::kRowElems } -> std::convertible_to<int>;
  { T::kDynamic } -> std::convertible_to<bool>;
};

// Marks a tensor as fixed at one row rather than riding the dynamic axis, and
// so staged once per call: e.g. the move-set graph's board inputs, the single
// position every candidate row shares.
template <typename Tensor>
struct Static : Tensor {
  static constexpr bool kDynamic = false;
};

struct SpatialInput {
  static constexpr const char* kName = "input_spatial";
  using Elem = float;
  static constexpr int kRowElems = 0;
  static constexpr bool kDynamic = true;
};

struct ScalarInput {
  static constexpr const char* kName = "input_scalar";
  using Elem = float;
  static constexpr int kRowElems = 0;
  static constexpr bool kDynamic = true;
};

// The per-move inputs also name, as kBatchSource, the MoveFeatureArrays field
// they are staged from, so the service stages them by iterating the list.
// Dtypes and widths are move_set_encoder.h's own, so an encoded candidate set
// copies in without conversion.
struct MoveLettersInput {
  static constexpr const char* kName = "move_letters";
  using Elem = int32_t;
  static constexpr int kRowElems = move_set::kMoveMaxPlaced;
  static constexpr bool kDynamic = true;
  static constexpr auto kBatchSource = &move_set::MoveFeatureArrays::letters;
};

struct MoveBlanksInput {
  static constexpr const char* kName = "move_blanks";
  using Elem = uint8_t;
  static constexpr int kRowElems = move_set::kMoveMaxPlaced;
  static constexpr bool kDynamic = true;
  static constexpr auto kBatchSource = &move_set::MoveFeatureArrays::blanks;
};

struct MoveSquaresInput {
  static constexpr const char* kName = "move_squares";
  using Elem = int32_t;
  static constexpr int kRowElems = move_set::kMoveMaxPlaced;
  static constexpr bool kDynamic = true;
  static constexpr auto kBatchSource = &move_set::MoveFeatureArrays::squares;
};

struct MoveTileMaskInput {
  static constexpr const char* kName = "move_tile_mask";
  using Elem = uint8_t;
  static constexpr int kRowElems = move_set::kMoveMaxPlaced;
  static constexpr bool kDynamic = true;
  static constexpr auto kBatchSource = &move_set::MoveFeatureArrays::tile_mask;
};

struct MoveScalarsInput {
  static constexpr const char* kName = "move_scalars";
  using Elem = float;
  static constexpr int kRowElems = move_set::kMoveScalars;
  static constexpr bool kDynamic = true;
  static constexpr auto kBatchSource = &move_set::MoveFeatureArrays::scalars;
};

// The activation TrtEvalService applies to each row of a head's raw output.
// It belongs to the head's descriptor, not to the code reading it.
enum class RowDecode : uint8_t { kIdentity, kSoftmax, kSigmoid };

// The exported output names are the training target names (training_targets.h),
// so the descriptors reference those rather than restate them.
struct WldOutput {
  static constexpr const char* kName = WldTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kWldFloats;
  static constexpr bool kDynamic = true;
  // Raw logits; consumers get [P(win), P(draw), P(loss)].
  static constexpr RowDecode kDecode = RowDecode::kSoftmax;
};

struct ScoreDiffOutput {
  static constexpr const char* kName = ScoreDiffTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kScoreDiffOutputFloats;  // [mean, std]
  static constexpr bool kDynamic = true;
  // Already the Gaussian's parameters; std is made positive in-graph.
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

// The four placement heads: logits over the kFootprintClasses move footprints
// (training/footprint.h), in export order (PLACEMENT_HEAD_NAMES in
// position_eval/model.py). They decode as kIdentity because the legality mask
// must be applied before the softmax and is not part of the graph, so each
// consumer (the Python loss, the dashboard, the .mset generator) applies the
// masked softmax itself.
struct OppNextMaskOutput {
  static constexpr const char* kName = OppNextPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kFootprintClasses;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

struct SelfNextMaskOutput {
  static constexpr const char* kName = SelfNextPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kFootprintClasses;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

struct OppWinMaskOutput {
  static constexpr const char* kName = OppWinPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kFootprintClasses;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

struct SelfWinMaskOutput {
  static constexpr const char* kName = SelfWinPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kFootprintClasses;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

// ---------- move-proposal tensors ----------------------------------------
//
// The move proposal model runs as two graphs
// (py/scribblez/move_set_eval/proposal_export.py): a `cache` graph once per
// turn and a `step` graph once per evidence iteration. Three handoff tensors
// pass from cache to step: the board token map, the global summary, and the
// per-move encodings. The step graph also reads a padded evidence set.
//
// agent/move_proposal_nets.h copies the handoff tensors between the two nets'
// host buffers and applies any activations itself, so these descriptors carry
// no RowDecode.
//
// Every handoff tensor's inner dimension is the trunk channel width C. The
// descriptors leave it model-decided (kRowElems 0) rather than restate an
// architecture constant; NeuralNetBase::channels() reports it. The layout check
// therefore cannot tie a cache/step pair to one C: the shared
// proposal_export_id fingerprint does.

// board (1, 225, C): the trunk token map, a cache output and a step input.
struct BoardHandoff {
  static constexpr const char* kName = "board";
  using Elem = float;
  static constexpr int kRowElems = 0;
  static constexpr bool kDynamic = true;  // wrapped Static in both specs
};

// g (1, 3C): the global summary, a cache output and a step input.
struct GHandoff {
  static constexpr const char* kName = "g";
  using Elem = float;
  static constexpr int kRowElems = 0;
  static constexpr bool kDynamic = true;  // wrapped Static in both specs
};

// move_enc (M, C): the per-move encodings, a cache output and a step input.
// The only handoff tensor riding the dynamic "moves" axis; its per-row width C
// is what channels() reads.
struct MoveEncHandoff {
  static constexpr const char* kName = "move_enc";
  using Elem = float;
  static constexpr int kRowElems = 0;
  static constexpr bool kDynamic = true;
};

// planes (M, 4 * kSlotsPerCell, 225): the four placement heads' footprint
// probabilities, softmaxed in-graph with the catch-all classes dropped. Each
// head is kSlotsPerCell board-shaped channels: anchored class (cell, slot) sits
// at channel head * kSlotsPerCell + slot, the predicted-channel layout of
// agent/evidence_staging.h.
//
// Only the cache graph emits planes. They feed the "predicted" half of a
// simmed candidate's evidence token, and nothing reads evidence-conditioned
// planes, so the step graph omits them and avoids an M x 11,700-float buffer.
struct PlanesOutput {
  static constexpr const char* kName = "planes";
  using Elem = float;
  static constexpr int kRowElems = kPlacementHeads * kSlotsPerCell * kBoardCells;
  static constexpr bool kDynamic = true;
};

// gain (M, 1): the proves-best expected gain, already non-negative (softplus
// in-graph).
struct GainOutput {
  static constexpr const char* kName = "gain";
  using Elem = float;
  static constexpr int kRowElems = 1;
  static constexpr bool kDynamic = true;
};

// The step graph's padded evidence width E and per-token layout. E is
// proposal_export.py's DEFAULT_MAX_EVIDENCE; the plane and scalar counts mirror
// agent/evidence_staging.h's kNumEvidencePlanes and kNumEvidenceScalars. They
// are restated here to keep this header free of the agent and sim headers.
// move_proposal_nets.cpp static_asserts that the two agree, and the loader's
// width check catches any drift from the exported graph.
inline constexpr int kMaxEvidence = 64;
// 117: observed + predicted footprint channels (4 heads x kSlotsPerCell each)
// plus the candidate's own kSlotsPerCell-channel footprint one-hot.
inline constexpr int kEvidencePlanes = (2 * kPlacementHeads + 1) * kSlotsPerCell;
inline constexpr int kEvidenceScalars = 11;

// The evidence inputs are static tensors of shape (1, E, ...). Folding E into
// the row keeps M the step graph's only dynamic axis. Each row is E times the
// per-token width, which is fixed except for ev_move_enc's model-decided C.
struct EvMoveEncInput {
  static constexpr const char* kName = "ev_move_enc";
  using Elem = float;
  static constexpr int kRowElems = 0;     // E * C, C model-decided
  static constexpr bool kDynamic = true;  // wrapped Static in the spec
};

struct EvObsPlanesInput {
  static constexpr const char* kName = "ev_obs_planes";
  using Elem = float;
  static constexpr int kRowElems = kMaxEvidence * kEvidencePlanes * kBoardCells;
  static constexpr bool kDynamic = true;  // wrapped Static in the spec
};

struct EvObsScalarsInput {
  static constexpr const char* kName = "ev_obs_scalars";
  using Elem = float;
  static constexpr int kRowElems = kMaxEvidence * kEvidenceScalars;
  static constexpr bool kDynamic = true;  // wrapped Static in the spec
};

struct EvMaskInput {
  static constexpr const char* kName = "ev_mask";
  using Elem = uint8_t;
  static constexpr int kRowElems = kMaxEvidence;
  static constexpr bool kDynamic = true;  // wrapped Static in the spec
};

// ---------- type list ----------------------------------------------------

// A type list of tensor descriptors, in the style of training_targets.h's
// TargetList.
template <TensorDescriptor... Ts>
struct TensorList {
  static constexpr std::size_t size = sizeof...(Ts);

  // Meaningful only when every member has a fixed width; a model-decided
  // width contributes 0. Used to size aux-output rows.
  static constexpr int total_row_elems = (0 + ... + Ts::kRowElems);
};

// ---------- version requirements ----------------------------------------

// An encoding-version gate: an ONNX metadata entry the loader requires at an
// exact value. A model fed rows from a different encoding version than it was
// trained on is silently off-distribution, since shapes and dtypes typically
// survive a version bump. The gate is the only thing that catches it.
struct VersionRequirement {
  const char* key;
  int required;
  // The value assumed for a model with no such entry.
  int absent_value;
};

// Both families consume board rows, so both gate on the board-row encoding
// version. Absent reads as 0.
inline constexpr VersionRequirement kInputEncodingRequirement = {"input_encoding_version",
                                                                 kInputEncodingVersion, 0};

// No absent allowance (-1 never matches): every move-set export stamps it.
inline constexpr VersionRequirement kMoveEncodingRequirement = {"move_encoding_version",
                                                                move_set::kMoveEncodingVersion, -1};

// ---------- the two model families --------------------------------------

// The position evaluation model: N independent positions per call, every
// tensor on the batch axis.
class PositionEvaluationSpec {
 public:
  static constexpr const char* kGraph = kGraphPositionEval;

  // A model with no `graph` entry is accepted as a position model: older
  // position exports lack it, and the layout check rejects any other graph.
  static constexpr bool kAcceptUntaggedGraph = true;

  // Names the dynamic axis in the plan cache's profile tag, e.g. "batch_256".
  static constexpr const char* kAxisTag = "batch";
  static constexpr int kDefaultMaxRows = 256;

  static constexpr const char* kChannelsTensor = nullptr;

  // The row count the plan is tuned for, clamped to max_rows. Callers mostly
  // submit full batches, so optimize for the maximum.
  static constexpr int kOptRows = std::numeric_limits<int>::max();

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement};

  using Inputs = TensorList<SpatialInput, ScalarInput>;
  using MoveInputs = TensorList<>;
  using Outputs = TensorList<WldOutput, ScoreDiffOutput>;
  using AuxOutputs =
    TensorList<OppNextMaskOutput, SelfNextMaskOutput, OppWinMaskOutput, SelfWinMaskOutput>;

  // `count` rows, each one position's [spatial | scalar] floats as
  // GameStateEncoder::encode_input() writes them.
  struct Batch {
    const float* rows;
    int count;
  };
};

// The move set evaluation model: one position's board row plus M encoded
// candidates per call, M the only dynamic axis. The architecture exists to
// amortize one board encoding across a whole candidate set, so a call is always
// one position; the exported graph fixes the board inputs at one row.
class MoveSetEvaluationSpec {
 public:
  static constexpr const char* kGraph = kGraphMoveSetEval;

  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "moves";
  static constexpr const char* kChannelsTensor = nullptr;

  // Sized to cover a realistic full move set in one chunk. Each extra chunk
  // pays another synchronous round trip and another board trunk pass, the very
  // work this architecture exists to pay once, and a smaller bound saves almost
  // no memory. Measured on the parity fixture at M=4000: 0.37 ms in one chunk,
  // 0.94 ms at max 1024, 1.98 ms at 512; device memory 46 MiB at 4096 against
  // 42 MiB at 256.
  static constexpr int kDefaultMaxRows = 4096;

  // Near a typical full move set.
  static constexpr int kOptRows = 512;

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement,
                                                     kMoveEncodingRequirement};

  using Inputs = TensorList<Static<SpatialInput>, Static<ScalarInput>, MoveLettersInput,
                            MoveBlanksInput, MoveSquaresInput, MoveTileMaskInput, MoveScalarsInput>;
  using MoveInputs = TensorList<MoveLettersInput, MoveBlanksInput, MoveSquaresInput,
                                MoveTileMaskInput, MoveScalarsInput>;
  // Deliberately the position model's heads, so an agent's EvalObjective ranks
  // alternatives the same way whichever family produced the value.
  using Outputs = TensorList<WldOutput, ScoreDiffOutput>;
  using AuxOutputs = TensorList<>;

  // One position's board row (as GameStateEncoder::encode_input() writes it)
  // and that position's candidates.
  struct Batch {
    const float* board_row;
    const move_set::MoveFeatureArrays* moves;
  };
};

// ---------- the move-proposal families ----------------------------------
//
// The move proposal model runs incrementally inside its deployment loop
// (docs/plans/sim_residual_feedback.md), so it is served as two specs: the
// per-turn cache graph and the per-evidence-iteration step graph. The two
// exports are tied together by a shared proposal_export_id fingerprint, which
// agent/move_proposal_nets.h validates. That class drives both through
// NeuralNet directly, not TrtEvalService, so neither spec has a Batch struct.
// They are served at FP32 by default (MoveProposalNets::Params); reduced
// precision has not been validated for the evidence-fusion graph's masked_fill
// and 4-D einsum.
//
// max_rows defaults differ by graph. The cache graph's planes output is 11,700
// floats per candidate, held on device and in pinned host memory at max_rows:
// 192 MB per side at 4096 against 48 MB at 1024. So the cache graph takes
// 1024, and an extra chunk for a rare >1024-candidate turn costs nothing next
// to the sims that turn runs. The step graph has no planes and only a few
// C-wide floats per row, and runs once per evidence iteration, so it keeps
// the move-set graph's 4096.

// The cache graph: one position's board row plus M candidates, in, and the
// handoff tensors plus the evidence-free predictions (wld, score_diff, planes),
// out. Its inputs are the move-set graph's, staged the same way.
class MoveProposalCacheSpec {
 public:
  static constexpr const char* kGraph = kGraphMoveProposalCache;
  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "moves";
  static constexpr int kDefaultMaxRows = 1024;
  static constexpr int kOptRows = 512;

  static constexpr const char* kChannelsTensor = MoveEncHandoff::kName;

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement,
                                                     kMoveEncodingRequirement};

  using Inputs = TensorList<Static<SpatialInput>, Static<ScalarInput>, MoveLettersInput,
                            MoveBlanksInput, MoveSquaresInput, MoveTileMaskInput, MoveScalarsInput>;
  using MoveInputs = TensorList<MoveLettersInput, MoveBlanksInput, MoveSquaresInput,
                                MoveTileMaskInput, MoveScalarsInput>;
  using Outputs = TensorList<Static<BoardHandoff>, Static<GHandoff>, MoveEncHandoff, WldOutput,
                             ScoreDiffOutput, PlanesOutput>;
  using AuxOutputs = TensorList<>;
};

// The step graph: the handoff tensors plus a padded evidence set, in, and the
// evidence-conditioned wld and score_diff plus the proves-best gain, out.
class MoveProposalStepSpec {
 public:
  static constexpr const char* kGraph = kGraphMoveProposalStep;
  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "moves";
  static constexpr int kDefaultMaxRows = 4096;
  static constexpr int kOptRows = 512;

  static constexpr const char* kChannelsTensor = MoveEncHandoff::kName;

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement,
                                                     kMoveEncodingRequirement};

  using Inputs =
    TensorList<Static<BoardHandoff>, Static<GHandoff>, MoveEncHandoff, Static<EvMoveEncInput>,
               Static<EvObsPlanesInput>, Static<EvObsScalarsInput>, Static<EvMaskInput>>;
  using MoveInputs = TensorList<>;
  using Outputs = TensorList<WldOutput, ScoreDiffOutput, GainOutput>;
  using AuxOutputs = TensorList<>;
};

// The aux outputs are the placement heads, one per placement target.
static_assert(PositionEvaluationSpec::AuxOutputs::size == kPlacementHeads);

}  // namespace nn
}  // namespace scribblez
