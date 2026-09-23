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
// Threading contract: one call at a time, from any thread. predict() blocks
// until the outputs are on the host, and calls must never overlap, since they
// share one execution context and one set of staging buffers. There is no
// thread affinity: the stream and buffers carry their device, so serialized
// calls from different threads are fine. EvalService's mutex relies on this.

namespace scribblez {
namespace nn {

struct NeuralNetParamsBase {
  std::string onnx_path;
  int cuda_device_id = 0;

  int max_rows = 0;  // NeuralNetParams<Spec> sets the family default

  // BF16 by default: it has FP32's exponent range, so activations that
  // overflow FP16 stay finite (docs/plans/fp16_safe_serving.md has the
  // measurements). Opt into FP16 only for a model known to fit its range.
  Precision precision = Precision::kBF16;
  uint64_t workspace_bytes = uint64_t{1} << 30;  // 1 GiB TensorRT scratch
  std::string mount_root = "/workspace/mount";   // root of the engine-plan cache

  // Build at TensorRT optimization level 0, which takes the first working
  // kernel per layer instead of timing tactics. A cold build drops from tens of
  // seconds to a few, but inference is much slower: for tests and quick checks,
  // not production agents. Cached separately from full-optimization plans.
  bool fast_build = false;

  // Copy the spec's aux outputs back to the host on every predict(). Off by
  // default so that only a consumer that reads them pays for the per-call
  // device-to-host copy. No effect for a spec with no aux outputs.
  bool copy_aux = false;

  // Two callers may share one loaded service (PositionEvalService::create())
  // only if all fields are equal, since every field shapes the engine or its
  // buffers. That includes copy_aux, which decides whether aux host buffers
  // exist at all, and fast_build, which selects a different cached plan.
  bool operator==(const NeuralNetParamsBase&) const = default;

  // Register the command-line subset of these fields. Call before parsing argv.
  void add_options(boost::program_options::options_description& desc);
};

// The base params with max_rows at the family's default.
template <typename Spec>
struct NeuralNetParams : NeuralNetParamsBase {
  NeuralNetParams() { max_rows = Spec::kDefaultMaxRows; }
};

// One engine I/O tensor as the runtime expects it: a spec descriptor
// (model_specs.h) flattened to runtime data. The loader checks each one against
// the model's own declarations.
struct TensorSpec {
  const char* name;
  std::size_t elem_size;
  int elems_per_row;  // 0 where the model's own declaration decides
  bool dynamic;       // rides the spec's dynamic row axis
  bool aux;           // host copy only under params.copy_aux
};

// A spec (model_specs.h) flattened to the plain data NeuralNetBase runs on.
// The spans point at NeuralNet<Spec>'s static tables.
struct RuntimeSpec {
  const char* graph;
  bool accept_untagged_graph;
  std::span<const VersionRequirement> versions;
  const char* axis_tag;
  int opt_rows;
  std::span<const TensorSpec> tensors;
  // The tensor whose per-row width is the trunk channel count C, which
  // channels() reports. Null when the spec exposes no such tensor.
  const char* channels_tensor;
};

// The spec-independent runtime: engine build, the plan cache, metadata gates,
// layout validation, and buffer management. It is compiled once and driven by
// a RuntimeSpec; NeuralNet<Spec> below adds only typed buffer access.
class NeuralNetBase {
 public:
  ~NeuralNetBase();

  NeuralNetBase(const NeuralNetBase&) = delete;
  NeuralNetBase& operator=(const NeuralNetBase&) = delete;

  // Load params.onnx_path into a ready engine. Every checkpoint of one
  // architecture shares a cached plan, so a cache hit costs a weight refit
  // rather than a full build. Throws unless the model declares the spec's graph
  // and every encoding version the spec requires. Call exactly once, before
  // predict().
  void load();

  int max_rows() const;

  // Valid after load(): the board-row widths the model consumes. Zero for a
  // graph with no board inputs, such as the move-proposal step graph.
  int spatial_planes() const;
  int scalar_floats() const;

  // Valid after load(): the trunk channel width C, or zero for a spec with no
  // channels_tensor.
  int channels() const;

  // Valid after load(): the model's input-encoding arm, from its ONNX
  // metadata. derive_input_spec() (agent/candidate_evaluator.h) checks it
  // against the input widths.
  bool opp_leave_input() const;

  // Run the staged inputs. Requires 1 <= num_rows <= max_rows(); static
  // tensors always carry one row.
  void predict(int num_rows);

  // The host buffer bound to `name`; NeuralNet<Spec>::host() is the typed way
  // in. Null for an aux output without params.copy_aux.
  void* host_ptr(const char* name) const;

 protected:
  NeuralNetBase(const NeuralNetParamsBase& params, const RuntimeSpec& spec);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

namespace detail {

// The spec's descriptor lists flattened into one table: inputs, outputs, then
// aux outputs.
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

  // The host buffer for one of the spec's tensors, e.g. host<SpatialInput>().
  // Write inputs before predict(); read outputs after it, undecoded. Aux-output
  // buffers require params.copy_aux.
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
