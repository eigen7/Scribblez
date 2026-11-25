#pragma once

#include "belief/leave_params.h"
#include "encoding/input_encoder.h"
#include "encoding/leaf_eval_input_encoder.h"
#include "encoding/leave_inference_input_encoder.h"
#include "encoding/square_fates.h"
#include "nn/onnx_metadata.h"
#include "training/game_end_targets.h"
#include "training/square_fate_targets.h"
#include "training/training_targets.h"

#include <array>
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
// trt_eval_service.cpp.
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

// The leaf eval model's continuation heads (docs/plans/subtree_bot.md, "The
// search network"): what follows the leaf, from the side of the player who
// just moved. A per-square output is plane-major, kBoardCells floats per
// plane, in the row's board frame. Points are in units of kScoreDiffInputScale.
struct OwnershipOutput {
  static constexpr const char* kName = OwnershipTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = kSquareFateClasses * kBoardCells;  // a plane per SquareFate
  static constexpr bool kDynamic = true;
  // Raw logits; a consumer softmaxes each square over its class planes.
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

struct SquarePointsOutput {
  static constexpr const char* kName = SquarePointsTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = 2 * kBoardCells;  // [mean plane, variance plane]
  static constexpr bool kDynamic = true;
  // The moments themselves; the variance is made positive in-graph.
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

struct GoingOutOutput {
  static constexpr const char* kName = GoingOutTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = GoingOutTarget::kClasses;  // the mover, the opponent, nobody
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kSoftmax;
};

struct EndPointsOutput {
  static constexpr const char* kName = EndPointsTarget::kName;
  using Elem = float;
  static constexpr int kRowElems = 2;  // [mean, variance]
  static constexpr bool kDynamic = true;
  static constexpr RowDecode kDecode = RowDecode::kIdentity;
};

// The four placement heads: logits over the kFootprintClasses move footprints
// (training/footprint.h), in export order (PLACEMENT_HEAD_NAMES in
// position_eval/model.py). They decode as kIdentity because the legality mask
// must be applied before the softmax and is not part of the graph, so each
// consumer (the Python loss, the dashboard) applies the
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

// leave_params (N, 29): the tilt of each row's leave prior, belief::LeaveParams'
// layout. Consumed raw by belief::LeaveSampler, so it carries no RowDecode.
struct LeaveParamsOutput {
  static constexpr const char* kName = "leave_params";
  using Elem = float;
  static constexpr int kRowElems = belief::LeaveParams::kFloats;
  static constexpr bool kDynamic = true;
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

  // Each member's kRowElems, in list order: per-head row widths for code that
  // walks a list's buffers at runtime.
  static constexpr std::array<int, size> row_elems = {Ts::kRowElems...};
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

// Every family that consumes board rows gates on the board-row encoding
// version. Absent reads as 0.
inline constexpr VersionRequirement kInputEncodingRequirement = {"input_encoding_version",
                                                                 kInputEncodingVersion, 0};

// ---------- the position evaluation family --------------------------------------

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

  // The row count the plan is tuned for, clamped to max_rows. Callers mostly
  // submit full batches, so optimize for the maximum.
  static constexpr int kOptRows = std::numeric_limits<int>::max();

  static constexpr VersionRequirement kVersions[] = {kInputEncodingRequirement};

  using Inputs = TensorList<SpatialInput, ScalarInput>;
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

// ---------- the leave inference family -----------------------------------
//
// The leave inference model (docs/plans/subtree_bot_v0.md, "Serving"): N
// positions' rows (encoding/leave_inference_input_encoder.h), each to its 29
// tilt floats. belief/leave_inference_net.h drives it through NeuralNet
// directly: its one output is consumed raw, so TrtEvalService's per-row decode
// would add nothing.
class LeaveInferenceSpec {
 public:
  static constexpr const char* kGraph = kGraphLeaveInference;
  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "batch";
  static constexpr const char* kChannelsTensor = nullptr;

  // The agent batches ~2,000 queries a turn; v0's one consumer, the survey
  // tool, submits one position per call, so the plan is tuned for that.
  static constexpr int kDefaultMaxRows = 256;
  static constexpr int kOptRows = 1;

  // No absent allowance: every export stamps it.
  static constexpr VersionRequirement kVersions[] = {
    {"leave_inference_format_version", kLeaveInferenceFormatVersion, -1}};

  using Inputs = TensorList<SpatialInput, ScalarInput>;
  using Outputs = TensorList<LeaveParamsOutput>;
  using AuxOutputs = TensorList<>;
};

// The DeepBot leaf evaluation model (py/scribblez/deep_bot_leaf_eval): a
// LeafEvalInputEncoder row to win/draw/loss and score difference, from the
// side of the player who moved last. Batches are large: a rollout kernel
// hands it every leaf of a batch at once.
class LeafEvalSpec {
 public:
  static constexpr const char* kGraph = kGraphDeepBotLeafEval;
  static constexpr bool kAcceptUntaggedGraph = false;
  static constexpr const char* kAxisTag = "batch";
  static constexpr const char* kChannelsTensor = nullptr;
  static constexpr int kDefaultMaxRows = 1024;
  static constexpr int kOptRows = 1024;
  // A transformer leaf's fused attention needs more than the default 1 GiB at
  // kDefaultMaxRows: at 1 GiB TensorRT finds no tactic for it at all, and at
  // 2 GiB it builds a plan that plays ~1.7x slower than this cap's.
  // py/scribblez/workloads/deep_bot_leaf_eval.py's LEAF_TRT_WORKSPACE_GB mirrors it.
  static constexpr uint64_t kWorkspaceBytes = uint64_t{8} << 30;
  static constexpr VersionRequirement kVersions[] = {
    {"leaf_eval_format_version", kLeafEvalFormatVersion, -1}};
  using Inputs = TensorList<SpatialInput, ScalarInput>;
  using Outputs = TensorList<WldOutput, ScoreDiffOutput, OwnershipOutput, SquarePointsOutput,
                             GoingOutOutput, EndPointsOutput>;
  using AuxOutputs = TensorList<>;
};

// The aux outputs are the placement heads, one per placement target.
static_assert(PositionEvaluationSpec::AuxOutputs::size == kPlacementHeads);

}  // namespace nn
}  // namespace scribblez
