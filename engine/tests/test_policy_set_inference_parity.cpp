// The policy-set model's policy form served through TensorRT, checked against
// the PyTorch reference on the same inputs.
//
// The graph takes the move-set graph's inputs plus a static opponent-rack row
// and a per-move equity column, and emits one logit per candidate. A
// mis-bound tensor, a chunk boundary that drops rows, or a refit that serves
// one checkpoint's weights for another each change the numbers.
//
// The fixture comes from py/scripts/policy_set/gen_parity_fixture.py. Run
// with no arguments, the binary generates it through the Python generator at
// SCRIBBLEZ_PY_DIR and skips if that fails (no torch/onnx). Pass a fixture
// directory as the first non-gtest argument to reuse one instead:
//   test_policy_set_inference_parity [<fixture_dir>]

#include "nn/eval_service.h"
#include "nn/neural_net.h"
#include "nn/trt_eval_service.h"
#include "parity_fixture.h"
#include "training/policy_set_inputs.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using scribblez::nn::PolicySetSpec;
using scribblez::nn::Precision;
using PolicyParams = scribblez::nn::NeuralNetParams<PolicySetSpec>;

namespace {

using scribblez::testing::generate_fixture;
using scribblez::testing::make_scratch_dir;
using scribblez::testing::read_binary;

// FP32 deviates from the reference only through kernel and reduction order.
// The BF16 run is a build/bind/decode smoke test of the production precision
// (test_mset_inference_parity.cpp explains why it may run FP32 kernels).
// Defects move logits by order-one amounts.
constexpr float kFp32Tol = 1e-3f;
constexpr float kBf16Tol = 0.1f;

// The fixture directory from the command line; empty means self-generate.
std::string g_fixture_dir;

class PolicySetInferenceParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    if (owns_cache_root()) std::filesystem::remove_all(cache_root_);
  }

  // A fresh scratch plan cache by default, so a test sees only its own plans.
  virtual std::filesystem::path make_cache_root() const {
    return make_scratch_dir("scribblez_policycache_");
  }
  virtual bool owns_cache_root() const { return true; }

  std::string model(const char* name) const { return dir_ + "/" + name; }
  PolicyParams params(const std::string& onnx_path, Precision precision, int max_moves) const;

  // Each fixture candidate's logit from the model at `onnx_path`. A
  // `max_moves` below the candidate count makes the service chunk the set.
  std::vector<float> run(const std::string& onnx_path, Precision precision, int max_moves);

  // Engine plans in this test's cache.
  int cached_plans() const;

  // Holds the worst deviation of `got` from `expected` to `tol`, printing it
  // under `label`; returns it.
  float expect_matches(const std::vector<float>& got, const std::vector<float>& expected,
                       const char* label, float tol) const;

  std::string dir_;
  std::filesystem::path generated_;  // non-empty iff this test generated the fixture
  std::filesystem::path cache_root_;

  std::vector<float> board_;
  std::vector<float> opp_rack_;
  scribblez::policy_set::CandidateInputs candidates_;
  std::vector<float> expected_a_;
  std::vector<float> expected_b_;
};

void PolicySetInferenceParityTest::SetUp() {
  cache_root_ = make_cache_root();
  if (!g_fixture_dir.empty()) {
    dir_ = g_fixture_dir;
    if (!std::filesystem::exists(dir_ + "/model_a.onnx")) GTEST_FAIL() << "no fixture in " << dir_;
  } else {
    generated_ = make_scratch_dir("scribblez_policyparity_");
    dir_ = generated_.string();
    if (!generate_fixture("scripts.policy_set.gen_parity_fixture", dir_)) {
      GTEST_SKIP() << "could not generate fixture; is torch/onnx installed?";
    }
  }
  board_ = read_binary<float>(dir_ + "/board.bin");
  opp_rack_ = read_binary<float>(dir_ + "/opp_rack.bin");
  scribblez::move_set::MoveFeatureArrays& f = candidates_.features;
  f.letters = read_binary<int32_t>(dir_ + "/move_letters.bin");
  f.blanks = read_binary<uint8_t>(dir_ + "/move_blanks.bin");
  f.squares = read_binary<int32_t>(dir_ + "/move_squares.bin");
  f.tile_mask = read_binary<uint8_t>(dir_ + "/move_tile_mask.bin");
  f.scalars = read_binary<float>(dir_ + "/move_scalars.bin");
  f.cross_cells = read_binary<int32_t>(dir_ + "/move_cross_cells.bin");
  f.cross_letters = read_binary<uint8_t>(dir_ + "/move_cross_letters.bin");
  f.count = int(f.scalars.size() / scribblez::move_set::kMoveScalars);
  candidates_.equity = read_binary<float>(dir_ + "/move_equity.bin");
  expected_a_ = read_binary<float>(dir_ + "/expected_a.bin");
  expected_b_ = read_binary<float>(dir_ + "/expected_b.bin");

  ASSERT_GT(f.count, 0);
  ASSERT_EQ(opp_rack_.size(), size_t(scribblez::policy_set::kOppRackFloats));
  ASSERT_EQ(candidates_.equity.size(), size_t(f.count));
  ASSERT_EQ(expected_a_.size(), size_t(f.count));
  ASSERT_EQ(expected_b_.size(), size_t(f.count));
}

PolicyParams PolicySetInferenceParityTest::params(const std::string& onnx_path, Precision precision,
                                                  int max_moves) const {
  PolicyParams p;
  p.onnx_path = onnx_path;
  p.precision = precision;
  p.max_rows = max_moves;
  p.mount_root = cache_root_.string();
  // Parity checks the inference plumbing, not kernel-tactic quality.
  p.fast_build = true;
  return p;
}

std::vector<float> PolicySetInferenceParityTest::run(const std::string& onnx_path,
                                                     Precision precision, int max_moves) {
  scribblez::nn::TrtEvalService<PolicySetSpec> service(params(onnx_path, precision, max_moves));
  service.load();
  EXPECT_EQ(board_.size(), size_t(service.spatial_planes()) * 225 + size_t(service.scalar_floats()))
    << "the fixture's board row is not the width the loaded model consumes";
  std::vector<float> logits(size_t(candidates_.count()));
  float* const head_out[] = {logits.data()};
  service.evaluate({board_.data(), opp_rack_.data(), &candidates_}, head_out);
  return logits;
}

int PolicySetInferenceParityTest::cached_plans() const {
  int plans = 0;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(cache_root_))
    plans += entry.path().extension() == ".engine";
  return plans;
}

float PolicySetInferenceParityTest::expect_matches(const std::vector<float>& got,
                                                   const std::vector<float>& expected,
                                                   const char* label, float tol) const {
  float worst = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    // A NaN would otherwise vanish from the max: every comparison with it is
    // false.
    const float deviation = std::abs(got[i] - expected[i]);
    if (!std::isfinite(deviation)) {
      ADD_FAILURE() << label << ": non-finite logit for move " << i << " = " << got[i];
      continue;
    }
    worst = std::max(worst, deviation);
  }
  std::cout << "  [" << label << "] max logit err = " << worst << " (tol " << tol << ")\n";
  EXPECT_LE(worst, tol) << label;
  return worst;
}

// For tests that do not inspect the cache: the persistent production plan
// cache, so later runs skip the build. Safe for the reasons
// test_mset_inference_parity.cpp's cached tests give: every hit is refitted,
// and fast_build plans live apart from production ones.
class PolicySetInferenceParityCachedTest : public PolicySetInferenceParityTest {
 protected:
  std::filesystem::path make_cache_root() const override {
    return scribblez::nn::NeuralNetParamsBase{}.mount_root;
  }
  bool owns_cache_root() const override { return false; }
};

TEST_F(PolicySetInferenceParityCachedTest, MatchesPyTorchReference) {
  const int m = candidates_.count();
  expect_matches(run(model("model_a.onnx"), Precision::kFP32, m), expected_a_, "FP32", kFp32Tol);
  expect_matches(run(model("model_a.onnx"), Precision::kBF16, m), expected_a_, "BF16", kBf16Tol);
}

// Every chunk re-sends the board and the opponent rack, and the last is short.
TEST_F(PolicySetInferenceParityCachedTest, ChunksACandidateSetLargerThanTheEngine) {
  const int chunk = 8;
  ASSERT_GT(candidates_.count(), chunk);
  ASSERT_NE(candidates_.count() % chunk, 0) << "the fixture must leave a short final chunk";
  expect_matches(run(model("model_a.onnx"), Precision::kFP32, chunk), expected_a_, "chunked",
                 kFp32Tol);
}

// B shares A's architecture, so it refits A's cached plan rather than building
// its own; only B's own reference shows the refit served B's weights.
TEST_F(PolicySetInferenceParityTest, RefitsOnePlanAcrossCheckpoints) {
  const int m = candidates_.count();
  expect_matches(run(model("model_a.onnx"), Precision::kFP32, m), expected_a_, "A", kFp32Tol);
  ASSERT_EQ(cached_plans(), 1);
  const std::vector<float> from_b = run(model("model_b.onnx"), Precision::kFP32, m);
  expect_matches(from_b, expected_b_, "B refitted", kFp32Tol);
  EXPECT_EQ(cached_plans(), 1) << "a same-architecture checkpoint must refit the cached plan";
  float apart = 0.0f;
  for (size_t i = 0; i < from_b.size(); ++i)
    apart = std::max(apart, std::abs(from_b[i] - expected_a_[i]));
  EXPECT_GT(apart, 100 * kFp32Tol) << "the two fixture models are too alike to detect a stale plan";
}

// A stale format version keeps every shape, so only the gate catches it.
TEST_F(PolicySetInferenceParityTest, RejectsAStaleFormatVersion) {
  std::string error;
  try {
    scribblez::nn::NeuralNet<PolicySetSpec>(
      params(model("model_stale.onnx"), Precision::kFP32, candidates_.count()))
      .load();
  } catch (const std::runtime_error& e) {
    error = e.what();
  }
  EXPECT_NE(error.find("policy_set_format_version"), std::string::npos) << error;
}

}  // namespace

// A custom main so the first non-gtest argument can name a fixture directory.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc >= 2) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
