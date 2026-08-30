#pragma once

#include "nn/model_specs.h"
#include "nn/trt_util.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace boost::program_options {
class options_description;
}

// A thin, synchronous wrapper around a TensorRT engine, specialized to a model
// family by its spec (model_specs.h).
//
// One net drives one engine from one thread AT A TIME: predict() blocks until
// the outputs are back, with no cross-thread batching and no async pipeline,
// and calls must never overlap. Serialized calls from different threads are
// fine -- the CUDA stream and buffers carry their device -- which is what
// both nn::EvalService's internal serialization and the generators'
// scorer-thread pattern rely on.

namespace scribblez {
namespace nn {

struct NeuralNetParamsBase {
  std::string onnx_path;  // exported ONNX model load() builds from
  int cuda_device_id = 0;

  int max_rows = 0;  // NeuralNetParams<Spec> sets the family default

  Precision precision = Precision::kFP16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache

  // Build at TensorRT optimization level 0: take the first working kernel per
  // layer instead of timing tactics. Cuts a cold build from tens of seconds to
  // a few, at the cost of much slower inference -- for tests and quick checks,
  // not production agents. Cached separately from full-optimization plans.
  bool fast_build = false;

  // Copy the spec's aux outputs back to host on every predict(), making their
  // host buffers valid. Off by default: aux outputs always stay bound on the
  // device, but only a consumer that reads them should pay their per-call
  // device-to-host copy. No-op for a spec with no aux outputs.
  bool copy_aux = false;

  // Register the command-line-facing subset, bound to this struct's fields.
  // Call before parsing argv.
  void add_options(boost::program_options::options_description& desc);
};

// The params for one model family: the base fields at the family's row-bound
// default.
template <typename Spec>
struct NeuralNetParams : NeuralNetParamsBase {
  NeuralNetParams() { max_rows = Spec::kDefaultMaxRows; }
};

// One engine I/O tensor as the runtime must see it -- a spec descriptor
// (model_specs.h) flattened to runtime data. The loader checks every field
// against the model's own declarations, pre-build on the parsed graph and
// post-deserialize on the engine.
struct TensorSpec {
  const char* name;
  std::size_t elem_size;
  int elems_per_row;  // 0 where the model's own declaration decides
  bool dynamic;       // rides the spec's dynamic row axis
  bool aux;           // host copy only under params.copy_aux
};

// Everything NeuralNetBase needs to serve one model family, as plain data.
// The spans point at NeuralNet<Spec>'s static tables.
struct RuntimeSpec {
  const char* graph;
  bool accept_untagged_graph;
  std::span<const VersionRequirement> versions;
  const char* axis_tag;
  int opt_rows;
  std::span<const TensorSpec> tensors;
  // The tensor whose per-row width is the trunk channel count C, read off after
  // load() for channels(); null for a family with no such handoff tensor (the
  // move-proposal specs set it, position/mset leave it null).
  const char* channels_tensor;
};

// All machinery -- engine build, the architecture-keyed refitted plan cache,
// metadata gates, layout validation, binding-table buffer management --
// compiled once here and driven by a RuntimeSpec; NeuralNet<Spec> below adds
// only typed access.
class NeuralNetBase {
 public:
  ~NeuralNetBase();

  NeuralNetBase(const NeuralNetBase&) = delete;
  NeuralNetBase& operator=(const NeuralNetBase&) = delete;

  // Build the engine from params.onnx_path, or deserialize a cached plan and
  // refit it with this model's weights -- every checkpoint of one architecture
  // shares a plan, keyed by architecture signature, precision, the spec's row
  // axis and bound, GPU compute capability, and TRT version. Throws unless the
  // model declares the spec's graph and every encoding version the spec
  // requires (see model_specs.h). Exactly once, before predict().
  void load();

  int max_rows() const;

  // Valid after load(): the board-row widths the served model consumes. Zero
  // for a graph that takes no board inputs (the move-proposal step graph, whose
  // board arrives pre-encoded as a handoff tensor).
  int spatial_planes() const;
  int scalar_floats() const;

  // Valid after load() for a spec that names a channels_tensor: the trunk
  // channel width C, read off that handoff tensor's per-row width. Zero for a
  // family that exposes no such tensor.
  int channels() const;

  // The model's input-encoding arm, from the ONNX metadata_props the exporter
  // stamps. Valid after load(); consumers cross-check it against the input
  // widths through input_encoder.h's registry.
  bool opp_leave_input() const;

  // Blocks until the outputs are back. Requires 1 <= num_rows <= max_rows().
  // Static tensors stage one row whatever num_rows is.
  void predict(int num_rows);

  // The host staging/readout buffer bound to `name`, which the engine is
  // guaranteed to expose -- NeuralNet<Spec>::host() is the typed way in. Null
  // for an aux output without params.copy_aux.
  void* host_ptr(const char* name) const;

 protected:
  NeuralNetBase(const NeuralNetParamsBase& params, const RuntimeSpec& spec);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace detail {

// The spec's tensor descriptor lists flattened into NeuralNetBase's runtime
// table, in list order: inputs, outputs, then aux outputs.
template <typename... In, typename... Out, typename... Aux>
constexpr std::array<TensorSpec, sizeof...(In) + sizeof...(Out) + sizeof...(Aux)> tensor_specs(
  TensorList<In...>, TensorList<Out...>, TensorList<Aux...>) {
  return {
    {TensorSpec{In::kName, sizeof(typename In::Elem), In::kRowElems, In::kDynamic, false}...,
     TensorSpec{Out::kName, sizeof(typename Out::Elem), Out::kRowElems, Out::kDynamic, false}...,
     TensorSpec{Aux::kName, sizeof(typename Aux::Elem), Aux::kRowElems, Aux::kDynamic, true}...}};
}

template <typename Spec>
inline constexpr auto kTensorSpecs =
  tensor_specs(typename Spec::Inputs{}, typename Spec::Outputs{}, typename Spec::AuxOutputs{});

}  // namespace detail

template <typename Spec>
class NeuralNet : public NeuralNetBase {
 public:
  explicit NeuralNet(const NeuralNetParams<Spec>& params) : NeuralNetBase(params, kRuntimeSpec) {}

  // The host buffer for one of the spec's tensors, e.g. host<SpatialInput>():
  // staging for inputs (write, then predict()), readout for outputs (valid
  // after predict(), raw logits). Aux-output buffers require params.copy_aux.
  template <typename Tensor>
  Tensor::Elem* host() {
    using TensorElem = Tensor::Elem;
    return static_cast<TensorElem*>(host_ptr(Tensor::kName));
  }
  template <typename Tensor>
  const Tensor::Elem* host() const {
    using TensorElem = Tensor::Elem;
    return static_cast<const TensorElem*>(host_ptr(Tensor::kName));
  }

 private:
  static constexpr RuntimeSpec kRuntimeSpec = {
    Spec::kGraph,   Spec::kAcceptUntaggedGraph, Spec::kVersions,      Spec::kAxisTag,
    Spec::kOptRows, detail::kTensorSpecs<Spec>, Spec::kChannelsTensor};
};

}  // namespace nn
}  // namespace scribblez
