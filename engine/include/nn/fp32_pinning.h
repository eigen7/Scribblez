#pragma once

// Pinning the overflow-prone region of an FP16 build to FP32, expressed over a
// minimal graph view so the walk is testable without a GPU or TensorRT. The
// TensorRT adapter (neural_net.cpp) implements Fp32PinGraph over an
// INetworkDefinition; tests implement it over an in-memory graph.

#include <span>
#include <string_view>

namespace scribblez {
namespace nn {

// A layer's role in the pinning walk.
enum class PinLayerKind {
  kOther,          // pin it, and propagate through its outputs
  kConstant,       // shape data: never matched, never traversed
  kRenormalizing,  // a scale/normalization whose output is back in FP16 range:
                   // pin it, but stop -- downstream is safe again
};

// The minimal view of a built network the pinning walk needs: layers addressed
// by index in [0, num_layers), tensors by an opaque stable id (any int that is
// equal iff the tensors are the same), and the two precision setters. Ids need
// not be dense or start at 0; the walk only compares them.
class Fp32PinGraph {
 public:
  virtual ~Fp32PinGraph() = default;

  virtual int num_layers() const = 0;
  virtual std::string_view layer_name(int layer) const = 0;
  virtual PinLayerKind layer_kind(int layer) const = 0;
  virtual int num_inputs(int layer) const = 0;
  virtual int input_tensor(int layer, int input) const = 0;
  virtual int num_outputs(int layer) const = 0;
  virtual int output_tensor(int layer, int output) const = 0;

  // Force the layer to compute in FP32 (setPrecision).
  virtual void pin_layer_fp32(int layer) = 0;
  // Force one of the layer's outputs to be stored in FP32 (setOutputType), so a
  // too-large value is not downcast between layers.
  virtual void pin_output_fp32(int layer, int output) = 0;
};

// Pin the overflow-prone region to FP32: every non-constant layer whose name
// contains one of `substrings`, then downstream through consumers (the shuffles
// and adds carrying the too-large values) up to and including the first
// renormalizing layer. Non-terminal pinned layers also force FP32 output
// storage. Returns the pinned layer count. Throws util::Exception if a
// substring matches no layer, or the walk runs away without renormalizing --
// either means the substring list (model_specs.h) has drifted from the exported
// architecture.
int pin_fp32_region(Fp32PinGraph& graph, std::span<const char* const> substrings);

}  // namespace nn
}  // namespace scribblez
