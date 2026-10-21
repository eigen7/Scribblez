// The leave inference model served through TensorRT (belief/leave_inference_net.h),
// checked against the PyTorch reference on the same rows, and its load gates:
// a model trained against other leave values, or at another format version,
// is refused.
//
// The fixture comes from py/scripts/leave_inference/gen_parity_fixture.py. Run
// with no arguments, the binary generates it through the Python generator at
// SCRIBBLEZ_PY_DIR and skips if that fails (no torch/onnx). Pass a fixture
// directory as the first non-gtest argument to reuse one instead:
//   test_leave_inference_parity [<fixture_dir>]

#include "belief/leave_inference_net.h"
#include "belief/leave_params.h"
#include "encoding/leave_inference_input_encoder.h"
#include "parity_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using scribblez::LeaveInferenceInputEncoder;
using scribblez::belief::LeaveInferenceNet;
using scribblez::belief::LeaveParams;
using scribblez::testing::generate_fixture;
using scribblez::testing::make_scratch_dir;
using scribblez::testing::read_binary;

namespace {

// TensorRT's FP32 still runs convolutions and matmuls in TF32, which leaves
// relative errors near 1e-4 (measured: theta 1.1e-4 absolute, a 8e-5 and b
// 1.8e-4 relative). a and b are checked relative to their own size, which
// spans orders of magnitude. A mis-bound tensor or a dropped chunk moves
// outputs by order-one amounts.
constexpr float kThetaTol = 5e-4f;
constexpr float kTiltRelTol = 1e-3f;

// The worst deviation of served LeaveParams from the reference rows: theta
// absolute, a and b relative.
struct Deviation {
  float theta = 0;
  float a = 0;
  float b = 0;
};

float relative_error(float got, float expected) {
  return std::abs(got - expected) / std::abs(expected);
}

Deviation worst_deviation(const std::vector<LeaveParams>& got, const std::vector<float>& expected) {
  constexpr int kA = scribblez::TILE_KINDS;
  Deviation worst;
  for (size_t i = 0; i < got.size(); ++i) {
    const float* e = expected.data() + i * LeaveParams::kFloats;
    for (int k = 0; k < kA; ++k) {
      worst.theta = std::max(worst.theta, std::abs(got[i].theta[k] - e[k]));
    }
    worst.a = std::max(worst.a, relative_error(got[i].a, e[kA]));
    worst.b = std::max(worst.b, relative_error(got[i].b, e[kA + 1]));
  }
  return worst;
}

// Smaller than the fixture's row count, so predict() runs several chunks.
constexpr int kMaxRows = 4;

std::string g_fixture_dir;

class LeaveInferenceParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    std::filesystem::remove_all(cache_root_);
  }

  LeaveInferenceNet::Params params(const char* model) const {
    LeaveInferenceNet::Params p;
    p.onnx_path = dir_ + "/" + model;
    p.max_rows = kMaxRows;
    p.fast_build = true;
    p.mount_root = cache_root_.string();
    return p;
  }

  // The message from constructing a net that must be refused.
  std::string refusal(const char* model, const std::string& leave_values) const;

  std::string dir_;
  std::filesystem::path generated_;  // non-empty iff this test generated the fixture
  std::filesystem::path cache_root_;
  std::string leave_values_;
};

void LeaveInferenceParityTest::SetUp() {
  cache_root_ = make_scratch_dir("scribblez_licache_");
  if (!g_fixture_dir.empty()) {
    dir_ = g_fixture_dir;
  } else {
    generated_ = make_scratch_dir("scribblez_liparity_");
    dir_ = generated_.string();
    if (!generate_fixture("scripts.leave_inference.gen_parity_fixture", dir_)) {
      GTEST_SKIP() << "could not generate fixture; is torch/onnx installed?";
    }
  }
  std::ifstream(dir_ + "/leave_values.txt") >> leave_values_;
}

std::string LeaveInferenceParityTest::refusal(const char* model,
                                              const std::string& leave_values) const {
  try {
    LeaveInferenceNet net(params(model), leave_values);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  ADD_FAILURE() << model << " should have been refused";
  return "";
}

TEST_F(LeaveInferenceParityTest, Fp32MatchesPyTorch) {
  const std::vector<float> rows = read_binary<float>(dir_ + "/inputs.bin");
  const std::vector<float> expected = read_binary<float>(dir_ + "/expected.bin");
  const int n = int(rows.size() / LeaveInferenceInputEncoder::kInputFloats);
  ASSERT_GT(n, kMaxRows);
  ASSERT_EQ(expected.size(), size_t(n) * LeaveParams::kFloats);

  LeaveInferenceNet net(params("model.onnx"), leave_values_);
  std::vector<LeaveParams> got(n);
  net.predict(rows.data(), n, got.data());

  const Deviation worst = worst_deviation(got, expected);
  std::cout << "  [fp32] theta abs err " << worst.theta << ", a rel err " << worst.a
            << ", b rel err " << worst.b << "\n";
  EXPECT_LE(worst.theta, kThetaTol);
  EXPECT_LE(worst.a, kTiltRelTol);
  EXPECT_LE(worst.b, kTiltRelTol);
}

TEST_F(LeaveInferenceParityTest, RefusesOtherLeaveValues) {
  const std::string msg = refusal("model.onnx", "other-leave-values");
  EXPECT_NE(msg.find(leave_values_), std::string::npos) << msg;
  EXPECT_NE(msg.find("other-leave-values"), std::string::npos) << msg;
}

TEST_F(LeaveInferenceParityTest, RefusesOtherFormatVersion) {
  const std::string msg = refusal("stale_format.onnx", leave_values_);
  EXPECT_NE(msg.find("leave_inference_format_version"), std::string::npos) << msg;
}

TEST_F(LeaveInferenceParityTest, RefusesAPublicModel) {
  const std::string msg = refusal("public.onnx", leave_values_);
  EXPECT_NE(msg.find("public"), std::string::npos) << msg;
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
