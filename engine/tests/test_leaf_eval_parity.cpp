// The DeepBot leaf eval model served through TensorRT (nn::LeafEvalSpec),
// checked against the PyTorch reference on the same rows, and its leave
// values gate.
//
// The fixture comes from py/scripts/deep_bot_leaf_eval/gen_parity_fixture.py.
// Run with no arguments, the binary generates it through the Python generator
// at SCRIBBLEZ_PY_DIR and skips if that fails (no torch/onnx). Pass a fixture
// directory as the first non-gtest argument to reuse one instead:
//   test_leaf_eval_parity [<fixture_dir>]

#include "encoding/leaf_eval_input_encoder.h"
#include "nn/model_specs.h"
#include "nn/neural_net.h"
#include "nn/onnx_metadata.h"
#include "parity_fixture.h"
#include "util/io.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using scribblez::LeafEvalInputEncoder;
using scribblez::testing::generate_fixture;
using scribblez::testing::make_scratch_dir;
using scribblez::testing::read_binary;
namespace nn = scribblez::nn;

namespace {

using LeafNet = nn::NeuralNet<nn::LeafEvalSpec>;

// Per row of expected.bin: the three logits, then the score difference's
// mean and spread.
constexpr int kExpectedFloats = nn::WldOutput::kRowElems + nn::ScoreDiffOutput::kRowElems;
// TensorRT's FP32 runs convolutions in TF32 (errors near 1e-4, as the other
// families measure); a mis-bound tensor moves outputs by order-one amounts.
constexpr float kFp32Tol = 2e-3f;
// Smaller than the fixture's row count, so the rows take several calls.
constexpr int kMaxRows = 4;

std::string g_fixture_dir;

class LeafEvalParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    std::filesystem::remove_all(cache_root_);
  }

  nn::NeuralNetParams<nn::LeafEvalSpec> params() const {
    nn::NeuralNetParams<nn::LeafEvalSpec> p;
    p.onnx_path = dir_ + "/model.onnx";
    p.max_rows = kMaxRows;
    p.precision = nn::Precision::kFP32;
    p.fast_build = true;
    p.mount_root = cache_root_.string();
    return p;
  }

  std::string dir_;
  std::filesystem::path generated_;  // non-empty iff this test generated the fixture
  std::filesystem::path cache_root_;
};

void LeafEvalParityTest::SetUp() {
  cache_root_ = make_scratch_dir("scribblez_leafcache_");
  if (!g_fixture_dir.empty()) {
    dir_ = g_fixture_dir;
  } else {
    generated_ = make_scratch_dir("scribblez_leafparity_");
    dir_ = generated_.string();
    if (!generate_fixture("scripts.deep_bot_leaf_eval.gen_parity_fixture", dir_)) {
      GTEST_SKIP() << "could not generate fixture; is torch/onnx installed?";
    }
  }
}

TEST_F(LeafEvalParityTest, Fp32MatchesPyTorch) {
  const std::vector<float> rows = read_binary<float>(dir_ + "/inputs.bin");
  const std::vector<float> expected = read_binary<float>(dir_ + "/expected.bin");
  const int n = int(rows.size() / LeafEvalInputEncoder::kInputFloats);
  ASSERT_GT(n, kMaxRows);
  ASSERT_EQ(expected.size(), size_t(n) * kExpectedFloats);

  LeafNet net(params());
  net.load();
  float worst = 0;
  for (int first = 0; first < n; first += kMaxRows) {
    const int count = std::min(kMaxRows, n - first);
    for (int i = 0; i < count; ++i)
      nn::stage_board_row(net, &rows[size_t(first + i) * LeafEvalInputEncoder::kInputFloats], i);
    net.predict(count);
    const float* wld = net.host<nn::WldOutput>();
    const float* score = net.host<nn::ScoreDiffOutput>();
    for (int i = 0; i < count; ++i) {
      const float* e = &expected[size_t(first + i) * kExpectedFloats];
      for (int k = 0; k < nn::WldOutput::kRowElems; ++k)
        worst = std::max(worst, std::abs(wld[nn::WldOutput::kRowElems * i + k] - e[k]));
      for (int k = 0; k < nn::ScoreDiffOutput::kRowElems; ++k) {
        const float got = score[nn::ScoreDiffOutput::kRowElems * i + k];
        worst = std::max(worst, std::abs(got - e[nn::WldOutput::kRowElems + k]));
      }
    }
  }
  std::cout << "  [fp32] worst abs err " << worst << "\n";
  EXPECT_LE(worst, kFp32Tol);
}

TEST_F(LeafEvalParityTest, RefusesOtherLeaveValues) {
  const std::string path = dir_ + "/model.onnx";
  std::string leave_values;
  std::ifstream(dir_ + "/leave_values.txt") >> leave_values;
  const nn::OnnxMetadata meta = nn::parse_onnx_metadata(scribblez::util::read_file_bytes(path));
  EXPECT_NO_THROW(nn::check_leave_values(meta, path, leave_values));
  try {
    nn::check_leave_values(meta, path, "other-leave-values");
    ADD_FAILURE() << "other leave values should have been refused";
  } catch (const std::exception& e) {
    EXPECT_NE(std::string(e.what()).find("other-leave-values"), std::string::npos) << e.what();
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
