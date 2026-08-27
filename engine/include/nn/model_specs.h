#pragma once

#include "encoding/input_encoder.h"
#include "nn/onnx_metadata.h"
#include "training/move_set_encoder.h"
#include "training/training_targets.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

// What each served model family IS, declared as data: its graph identity, its
// dynamic-axis sizing, the encoding versions it must match, and -- the one
// difference that makes the families distinct -- its input/output tensors as
// type lists of descriptors. nn::NeuralNet<Spec> turns a spec into a TensorRT
// runtime and nn::TrtEvalService<Spec> into an evaluation service; neither
// contains a family-specific branch, so adding a model family means writing a
// spec here, not another runtime. A family served through TrtEvalService adds
// its Batch staging overloads in trt_eval_service.cpp too; the move-proposal
// specs skip that -- their handoff outputs do not fit the service's row-uniform
// decode, so they are driven through NeuralNet directly by
// agent/move_proposal_runtime.h.
//
// Carries no CUDA/TensorRT dependency: GPU-free consumers (agent interfaces,
// test stubs) read specs freely.

namespace scribblez {
namespace nn {

// ---------- tensor descriptors ------------------------------------------

// One engine I/O tensor as the runtime stages it. A descriptor declares:
//   - kName: the exported graph's tensor name -- the contract string tying the
//     descriptor to a Python-side head/input and an ONNX tensor
//   - Elem: the element type host buffers are cast to
//   - kRowElems: the row width the staging and decode code is written at
//     (0 where the model's own declaration decides, e.g. the board widths the
//     encoding arm implies)
//   - kDynamic: whether the tensor rides the spec's dynamic row axis
// The loader validates all of it against the model's own declarations -- a
// model that disagrees would otherwise overrun buffers rather than fail.
template <typename T>
concept TensorDescriptor = requires {
  { T::kName } -> std::convertible_to<const char*>;
  typename T::Elem;
  { T::kRowElems } -> std::convertible_to<int>;
  { T::kDynamic } -> std::convertible_to<bool>;
};

// Marks a tensor as NOT riding the dynamic row axis: the engine declares it at
// one fixed row, staged once per call (the move-set graph's board inputs -- the
// P=1 position every candidate row shares).
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

// The per-move inputs additionally name the move_set::MoveFeatureArrays field
// their rows are staged from, so the service's staging loop is derived from
// the list rather than hand-written per tensor. Dtypes and widths are
// move_set_encoder.h's own, so an encoded candidate set copies in without
// conversion.
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

// How one row of an output head's raw floats becomes the values consumers
// get: the head's activation belongs to the spec, not to the service reading
// it. Every output descriptor declares one.
enum class RowDecode : uint8_t { kIdentity, kSoftmax, kSigmoid };

// The exporters' output names are the target names of training_targets.h
// (served to Python over the FFI), so the output descriptors reference them
// rather than restate them.
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

// The four placement-mask heads -- one float per board cell, raw logits -- in
// export order (onnx_export.py's MASK_HEAD_NAMES), also the SimObservation
// plane order and the .mset plane order.
struct OppNextMaskOutput {
  static constexpr const char* kName = OppNextPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kBoardCells;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kSigmoid;
};

struct SelfNextMaskOutput {
  static constexpr const char* kName = SelfNextPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kBoardCells;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kSigmoid;
};

struct OppWinMaskOutput {
  static constexpr const char* kName = OppWinPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kBoardCells;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kSigmoid;
};

struct SelfWinMaskOutput {
  static constexpr const char* kName = SelfWinPlacementTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kBoardCells;
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kSigmoid;
};

// The number of placement-plane heads (targets.PLANE_NAMES) -- the four
// *MaskOutput descriptors above, and the width of the move-proposal graphs'
// dense `planes` output.
inline constexpr int kNumPlacementPlanes = 4;

// ---------- move-proposal (evidence-path) tensors ------------------------
//
// The move proposal model (roadmap item 3) runs as two graphs: a per-turn
// `cache` graph and a per-evidence-iteration `step` graph
// (py/scribblez/move_set_eval/proposal_export.py). Three tensors pass between
// them -- the board token map, the global summary, and the per-move encodings
// -- and the step graph additionally reads a padded evidence set. NeuralNet
// stages these raw: the orchestrator (agent/move_proposal_runtime.h) reads the
// handoff tensors off the cache's host buffers, copies them into the step's
// input buffers, and applies each head's activation itself, so these carry
// their real (undecoded) layout, not a RowDecode the service would apply.

// The trunk channel width C the model decided is the inner dimension of every
// handoff tensor. Left model-decided (kRowElems 0), so the descriptors do not
// restate an architecture constant; the loader reads C off a handoff binding
// (NeuralNetBase::channels()) and the shared proposal_export_id fingerprint,
// not the layout check, is what pins a cache/step pair to one C.

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

// planes (M, 4, 225): the four placement-plane heads' per-cell logits, a raw
// output of both graphs. No kDecode: unlike the mask/wld/score_diff descriptors,
// these two are read only by the orchestrator (which sigmoids the planes and
// reads the softplus'd gain as is), never by TrtEvalService's RowDecode path, so
// a decode field here would be dead metadata.
struct PlanesOutput {
  static constexpr const char* kName = "planes";
  using Elem = float;
  static constexpr int kRowElems = kNumPlacementPlanes * kBoardCells;
  static constexpr bool kDynamic = true;
};

// gain (M, 1): the proves-best expected gain, softplus'd in-graph (>= 0), so a
// step output the orchestrator reads as is (no kDecode; see PlanesOutput).
struct GainOutput {
  static constexpr const char* kName = "gain";
  using Elem = float;
  static constexpr int kRowElems = 1;
  static constexpr bool kDynamic = true;
};

// The step graph's padded evidence width E and per-token layout. E is
// proposal_export.py's DEFAULT_MAX_EVIDENCE; the plane/scalar counts mirror
// agent/evidence_staging.h's kNumEvidencePlanes / kNumEvidenceScalars (and,
// through it, evidence_fusion.py). Restated here -- as the graph-name strings
// are -- so model_specs.h stays free of the heavy agent/sim headers;
// move_proposal_runtime.cpp static_asserts the two agree, and the loader's
// tensor-width check fails loudly on any drift against the exported graph.
inline constexpr int kMaxEvidence = 64;
inline constexpr int kEvidencePlanes = 9;
inline constexpr int kEvidenceScalars = 11;

// The evidence inputs are fixed-width leading-1 batches (1, E, ...): E folded
// into the row keeps M the step graph's only dynamic axis. Each carries a
// compile-time row width E * rest, except ev_move_enc, whose rest is the
// model-decided C (kRowElems 0, like the handoff tensors).
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

// The training_targets.h TargetList idiom over tensor descriptors.
template <TensorDescriptor... Ts>
struct TensorList {
  static constexpr std::size_t size = sizeof...(Ts);

  // Meaningful only for lists whose every member fixes its width (aux-output
  // row sizing); a model-decided width contributes 0.
  static constexpr int total_row_elems = (0 + ... + Ts::kRowElems);
};

// ---------- version requirements ----------------------------------------

// One ONNX metadata entry the loader must find at an exact value: the
// encoding-version gates. A checkpoint trained under one encoding version fed
// rows from another would be silently off-distribution -- tensor shapes and
// dtypes all survive a version bump -- so the gate is the only thing standing
// between a stale model and wrong evaluations.
struct VersionRequirement {
  const char* key;
  int required;
  // What an export that predates the entry reads as (see onnx_metadata.h).
  int absent_value;
};

// Both families consume board rows, so both gate on the board-row encoding
// version. Absent reads as 0.
inline constexpr VersionRequirement kInputEncodingRequirement = {"input_encoding_version",
                                                                 kInputEncodingVersion, 0};

// The move-feature encoding gate has no absent allowance: every move-set
// export has stamped it since the family existed.
inline constexpr VersionRequirement kMoveEncodingRequirement = {"move_encoding_version",
                                                                move_set::kMoveEncodingVersion, -1};

// ---------- the two model families --------------------------------------

// The position evaluation model: N independent positions per call, every
// tensor riding the batch axis.
class PositionEvaluationSpec {
 public:
  static constexpr const char* kGraph = kGraphPositionEval;

  // An export predating the `graph` entry names nothing and is accepted: the
  // position graph is the only one that existed then (every position
  // checkpoint on the mount is such an export), and the runtime's tensor
  // layout checks reject anything else at load anyway.
  static constexpr bool kAcceptUntaggedGraph = true;

  // Names the dynamic axis in the engine-plan cache's profile tag
  // ("batch_256"): a plan is only valid within the row bounds it was built
  // for, and the two families size different axes.
  static constexpr const char* kAxisTag = "batch";
  static constexpr int kDefaultMaxRows = 256;

  // No trunk-channel handoff tensor: only the move-proposal specs expose C.
  static constexpr const char* kChannelsTensor = nullptr;

  // The row count the plan is optimized for, clamped to the params' max_rows:
  // agents fill whole batches, so OPT = MAX.
  static constexpr int kOptRows = std::numeric_limits<int>::max();

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement};

  using Inputs = TensorList<SpatialInput, ScalarInput>;
  using MoveInputs = TensorList<>;
  using Outputs = TensorList<WldOutput, ScoreDiffOutput>;
  using AuxOutputs =
    TensorList<OppNextMaskOutput, SelfNextMaskOutput, OppWinMaskOutput, SelfWinMaskOutput>;

  // count rows, each one position's [spatial | scalar] floats as
  // GameStateEncoder::encode_input() writes them.
  struct Batch {
    const float* rows;
    int count;
  };
};

// The move set evaluation model: one position's board row plus M encoded
// candidates per call, M the only dynamic axis. Amortizing one board encode
// across a whole candidate set is the architecture's reason to exist, so a
// call is a position and its candidates, never a batch of positions -- the
// exported graph is the P=1 specialization, its board inputs static.
class MoveSetEvaluationSpec {
 public:
  static constexpr const char* kGraph = kGraphMoveSetEval;

  // Every move-set export has named its graph since the family existed.
  static constexpr bool kAcceptUntaggedGraph = false;

  static constexpr const char* kAxisTag = "moves";

  // No trunk-channel handoff tensor: only the move-proposal specs expose C.
  static constexpr const char* kChannelsTensor = nullptr;

  // Upper bound on the candidates one predict() scores. Sized to the realistic
  // move-set ceiling rather than trimmed: every chunk past the first re-pays a
  // full synchronous round trip and another board trunk pass -- the
  // per-position work this architecture exists to pay once -- while buying
  // back almost no memory. Measured on the parity fixture at M=4000: 0.37 ms
  // in one chunk, 0.94 ms at max 1024, 1.98 ms at 512, against device memory
  // of 46 MiB at 4096 and 42 MiB at 256.
  static constexpr int kDefaultMaxRows = 4096;

  // The middle of the profile, near a typical full move set.
  static constexpr int kOptRows = 512;

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement,
                                                     kMoveEncodingRequirement};

  using Inputs = TensorList<Static<SpatialInput>, Static<ScalarInput>, MoveLettersInput,
                            MoveBlanksInput, MoveSquaresInput, MoveTileMaskInput, MoveScalarsInput>;
  using MoveInputs = TensorList<MoveLettersInput, MoveBlanksInput, MoveSquaresInput,
                                MoveTileMaskInput, MoveScalarsInput>;
  // Deliberately the position spec's scoring heads: an agent's EvalObjective
  // ranks alternatives identically whichever family produced the value -- the
  // families differ in how a value is obtained, not in what it means.
  using Outputs = TensorList<WldOutput, ScoreDiffOutput>;
  using AuxOutputs = TensorList<>;

  // One position's board row (as GameStateEncoder::encode_input() writes it)
  // and that position's candidates.
  struct Batch {
    const float* board_row;
    const move_set::MoveFeatureArrays* moves;
  };
};

// ---------- the move-proposal (evidence-path) families -------------------
//
// The move proposal model's deployment loop (docs/sim_residual_feedback.md)
// runs it incrementally as two graphs, so it is served as two specs, not one:
// a per-turn `cache` graph and a per-evidence-iteration `step` graph, tied by
// the shared proposal_export_id fingerprint the orchestrator validates. Both
// are driven directly through NeuralNet<Spec> by agent/move_proposal_runtime.h
// -- NOT through TrtEvalService, whose row-uniform decode loop cannot serve the
// static (board/g) or raw (move_enc) handoff outputs -- so neither carries a
// Batch struct or a TrtEvalService instantiation. Served at FP32 for item 3
// (the runtime defaults its params so); the fusion graph's masked_fill / 4D
// einsum make FP16 a later, separately gated optimization.

// The cache graph: one position's board row plus M candidates -> the reusable
// cache (board/g/move_enc) and the evidence-free predictions (wld/score_diff/
// planes). M is the only dynamic axis; board inputs are the mset graph's, so a
// cache call stages exactly as a move-set call does.
class MoveProposalCacheSpec {
 public:
  static constexpr const char* kGraph = kGraphMoveProposalCache;
  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "moves";
  static constexpr int kDefaultMaxRows = 4096;
  static constexpr int kOptRows = 512;

  // The handoff tensor whose per-row width is the trunk channels C.
  static constexpr const char* kChannelsTensor = MoveEncHandoff::kName;

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement,
                                                     kMoveEncodingRequirement};

  using Inputs = TensorList<Static<SpatialInput>, Static<ScalarInput>, MoveLettersInput,
                            MoveBlanksInput, MoveSquaresInput, MoveTileMaskInput, MoveScalarsInput>;
  using MoveInputs = TensorList<MoveLettersInput, MoveBlanksInput, MoveSquaresInput,
                                MoveTileMaskInput, MoveScalarsInput>;
  // board/g are leading-1 static; move_enc and the prediction heads ride the
  // "moves" axis.
  using Outputs = TensorList<Static<BoardHandoff>, Static<GHandoff>, MoveEncHandoff, WldOutput,
                             ScoreDiffOutput, PlanesOutput>;
  using AuxOutputs = TensorList<>;
};

// The step graph: the cache tensors plus a padded width-E evidence set -> the
// evidence-conditioned wld/score_diff/planes and the proves-best gain. M
// ("moves") is the only dynamic axis; board/g are static handoff inputs,
// move_enc rides "moves", and the evidence inputs are fixed-width leading-1
// batches.
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
  using Outputs = TensorList<WldOutput, ScoreDiffOutput, PlanesOutput, GainOutput>;
  using AuxOutputs = TensorList<>;
};

// The position model's aux-output count, for consumers sizing mask planes
// without naming the list.
inline constexpr int kNumMaskHeads = PositionEvaluationSpec::AuxOutputs::size;

}  // namespace nn
}  // namespace scribblez
