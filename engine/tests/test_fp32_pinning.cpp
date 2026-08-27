// Unit tests for the FP32-pinning walk (nn/fp32_pinning.h) over an in-memory
// Fp32PinGraph -- no TensorRT, no GPU, no model fixture. The walk is the guard
// that keeps an FP16 build's overflow-prone region in FP32; these pin its own
// behavior (match, propagate, stop, skip) and its two drift alarms (a substring
// that matches nothing, and a walk that never renormalizes).

#include "nn/fp32_pinning.h"
#include "util/exception.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace scribblez::nn {
namespace {

// A layer in the fake graph. Tensor ids are plain ints the test assigns; by
// convention the input feeds tensor 0 and layer i emits tensor i + 1.
struct FakeLayer {
  std::string name;
  PinLayerKind kind = PinLayerKind::kOther;
  std::vector<int> inputs;
  std::vector<int> outputs;
};

class FakeGraph : public Fp32PinGraph {
 public:
  explicit FakeGraph(std::vector<FakeLayer> layers) : layers_(std::move(layers)) {}

  int num_layers() const override { return int(layers_.size()); }
  std::string_view layer_name(int layer) const override { return layers_[size_t(layer)].name; }
  PinLayerKind layer_kind(int layer) const override { return layers_[size_t(layer)].kind; }
  int num_inputs(int layer) const override { return int(layers_[size_t(layer)].inputs.size()); }
  int num_outputs(int layer) const override { return int(layers_[size_t(layer)].outputs.size()); }
  int input_tensor(int layer, int input) const override {
    return layers_[size_t(layer)].inputs[size_t(input)];
  }
  int output_tensor(int layer, int output) const override {
    return layers_[size_t(layer)].outputs[size_t(output)];
  }
  void pin_layer_fp32(int layer) override { pinned_layers.push_back(layer); }
  void pin_output_fp32(int layer, int output) override {
    pinned_outputs.emplace_back(layer, output);
  }

  bool layer_pinned(int layer) const {
    return std::find(pinned_layers.begin(), pinned_layers.end(), layer) != pinned_layers.end();
  }

  std::vector<int> pinned_layers;
  std::vector<std::pair<int, int>> pinned_outputs;

 private:
  std::vector<FakeLayer> layers_;
};

const char* const kPoolFc[] = {"pool_fc"};

TEST(Fp32Pinning, MatchesSeedAndPropagatesToRenormalizing) {
  FakeGraph g({
    {"trunk/pool_fc/gemm", PinLayerKind::kOther, {0}, {1}},
    {"trunk/add", PinLayerKind::kOther, {1}, {2}},
    {"trunk/norm", PinLayerKind::kRenormalizing, {2}, {3}},
    {"head/gemm", PinLayerKind::kOther, {3}, {4}},
  });
  const int pinned = pin_fp32_region(g, kPoolFc);
  EXPECT_EQ(pinned, 3);
  EXPECT_TRUE(g.layer_pinned(0));
  EXPECT_TRUE(g.layer_pinned(1));
  EXPECT_TRUE(g.layer_pinned(2));   // the renormalizing layer is pinned...
  EXPECT_FALSE(g.layer_pinned(3));  // ...but the walk stops there.
  // Non-terminal pinned layers force FP32 output storage; the terminal
  // renormalizing layer does not.
  const std::vector<std::pair<int, int>> want_outputs{{0, 0}, {1, 0}};
  EXPECT_EQ(g.pinned_outputs, want_outputs);
}

TEST(Fp32Pinning, ThrowsWhenNoLayerMatches) {
  FakeGraph g({{"trunk/gemm", PinLayerKind::kOther, {0}, {1}}});
  EXPECT_THROW(pin_fp32_region(g, kPoolFc), util::Exception);
}

TEST(Fp32Pinning, DoesNotMatchConstantLayers) {
  // The only name containing the substring is a constant, so it is not a match.
  FakeGraph g({{"trunk/pool_fc/const", PinLayerKind::kConstant, {}, {1}}});
  EXPECT_THROW(pin_fp32_region(g, kPoolFc), util::Exception);
}

TEST(Fp32Pinning, SkipsConstantConsumersDuringPropagation) {
  FakeGraph g({
    {"trunk/pool_fc/gemm", PinLayerKind::kOther, {0}, {1}},
    {"trunk/shape_const", PinLayerKind::kConstant, {1}, {2}},  // consumes the seed's output
    {"trunk/norm", PinLayerKind::kRenormalizing, {1}, {3}},    // also consumes it
  });
  const int pinned = pin_fp32_region(g, kPoolFc);
  EXPECT_EQ(pinned, 2);  // seed + the renormalizing layer; the constant is skipped
  EXPECT_FALSE(g.layer_pinned(1));
  EXPECT_TRUE(g.layer_pinned(2));
}

TEST(Fp32Pinning, PinsASharedDownstreamLayerOnce) {
  // Diamond: the seed's output feeds two branches that rejoin at one layer.
  FakeGraph g({
    {"trunk/pool_fc/gemm", PinLayerKind::kOther, {0}, {1}},
    {"trunk/branch_a", PinLayerKind::kOther, {1}, {2}},
    {"trunk/branch_b", PinLayerKind::kOther, {1}, {3}},
    {"trunk/join", PinLayerKind::kRenormalizing, {2, 3}, {4}},
  });
  const int pinned = pin_fp32_region(g, kPoolFc);
  EXPECT_EQ(pinned, 4);
  EXPECT_EQ(std::count(g.pinned_layers.begin(), g.pinned_layers.end(), 3), 1);
}

TEST(Fp32Pinning, ThrowsWhenTheWalkRunsAwayWithoutRenormalizing) {
  std::vector<FakeLayer> layers;
  layers.push_back({"trunk/pool_fc/gemm", PinLayerKind::kOther, {0}, {1}});
  for (int i = 1; i < 70; ++i)
    layers.push_back({"trunk/op" + std::to_string(i), PinLayerKind::kOther, {i}, {i + 1}});
  FakeGraph g(std::move(layers));
  EXPECT_THROW(pin_fp32_region(g, kPoolFc), util::Exception);
}

}  // namespace
}  // namespace scribblez::nn
