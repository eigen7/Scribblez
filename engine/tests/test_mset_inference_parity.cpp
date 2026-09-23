// The move-set evaluation model served through TensorRT, checked against the
// PyTorch reference on the same inputs.
//
// This model takes seven input tensors across three dtypes and scores a whole
// candidate set in one pass, so more can go silently wrong than in the position
// runtime: a mis-sized binding, a move tensor bound to the wrong device
// pointer, a chunk boundary that drops rows, or a plan-cache hit that serves
// one checkpoint's weights for another. Each of these changes the numbers.
//
// The fixture comes from py/scripts/move_set_eval/gen_mset_parity_fixture.py:
// two same-architecture random-init models, plus variants that the loader must
// reject, one board row, and one candidate set. Run with no arguments, the
// binary generates it through the Python generator at SCRIBBLEZ_PY_DIR and
// skips if that fails (no torch/onnx). Pass a fixture directory as the first
// non-gtest argument to reuse one instead:
//   test_mset_inference_parity [<fixture_dir>]

#include "nn/eval_service.h"
#include "nn/neural_net.h"
#include "nn/trt_eval_service.h"
#include "training/move_set_encoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using scribblez::move_set::MoveFeatureArrays;
using scribblez::nn::MoveSetEvaluationSpec;
using scribblez::nn::PositionEvaluationSpec;
using scribblez::nn::Precision;
using MsetParams = scribblez::nn::NeuralNetParams<MoveSetEvaluationSpec>;
using PositionParams = scribblez::nn::NeuralNetParams<PositionEvaluationSpec>;

namespace {

// Fields per row of expected_*.bin: win_prob, p_win, p_draw, p_loss,
// score_diff_mean, score_diff_std.
constexpr int kFieldsPerRow = 6;

// Allowed deviation from the reference, per field group: the four
// probabilities, and the score-diff pair, which is in points.
struct Tolerance {
  float prob;
  float score_diff;
};

// FP32 deviates from the reference only through kernel and reduction order;
// FP16 also through its own precision, over an attention softmax across 225
// board tokens. Real defects (a mis-bound tensor, the wrong checkpoint's
// weights) move outputs by order-one amounts, far outside these bounds. The
// test prints the actual deviations, for retuning if a future model
// legitimately needs more slack.
//
// A reduced-precision flag only permits 16-bit kernels; whether TensorRT picks
// them depends on unrelated builder settings. Requesting a refittable plan, for
// example, can make it choose FP32 throughout, and the 16-bit run then
// reproduces FP32 bit for bit. As currently configured, FP16 does get 16-bit
// kernels (deviation ~6e-05), so that case tests FP16 arithmetic. BF16 does
// not: its run matches FP32 bit for bit, so the BF16 case is a
// build/bind/decode smoke test of the production precision. BF16 accuracy on
// real models is covered in docs/plans/fp16_safe_serving.md.
constexpr Tolerance kFp32Tol{1e-4f, 0.01f};
constexpr Tolerance kFp16Tol{5e-3f, 0.2f};
constexpr Tolerance kBf16Tol{1e-2f, 0.5f};  // 8 mantissa bits to FP16's 10

// The fixture directory from the command line; empty means self-generate.
std::string g_fixture_dir;

template <typename T>
std::vector<T> read_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    ADD_FAILURE() << "cannot open " << path;
    return {};
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<T> out(size_t(bytes) / sizeof(T));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

std::filesystem::path make_scratch_dir(const char* prefix) {
  std::filesystem::path base =
    std::filesystem::temp_directory_path() /
    (prefix + std::to_string(::time(nullptr)) + "_" + std::to_string(std::random_device{}()));
  std::filesystem::create_directories(base);
  return base;
}

// False if the generator is unavailable or fails (for example, torch or onnx
// is not installed).
bool generate_fixture(const std::string& out_dir) {
#ifdef SCRIBBLEZ_PY_DIR
  const std::string py_dir = SCRIBBLEZ_PY_DIR;
  const std::string cmd = "cd \"" + py_dir + "\" && PYTHONPATH=\"" + py_dir +
                          "\" python3 -m scripts.move_set_eval.gen_mset_parity_fixture "
                          "--out-dir \"" +
                          out_dir + "\"";
  return std::system(cmd.c_str()) == 0;
#else
  (void)out_dir;
  return false;
#endif
}

// The error message from loading `params`, which must fail. Records a test
// failure and returns "" if the load succeeds.
template <typename Spec>
std::string load_failure_message(const scribblez::nn::NeuralNetParams<Spec>& params) {
  try {
    scribblez::nn::NeuralNet<Spec>(params).load();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  ADD_FAILURE() << "loading " << params.onnx_path << " should have been rejected";
  return "";
}

// The fixture and an engine-plan cache, set up per test. By default the cache
// is a fresh scratch directory, so a test sees only the plans it built.
class MsetInferenceParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    if (!cache_root_.empty() && owns_cache_root()) std::filesystem::remove_all(cache_root_);
  }

  virtual std::filesystem::path make_cache_root() const {
    return make_scratch_dir("scribblez_msetcache_");
  }
  virtual bool owns_cache_root() const { return true; }

  std::string model(const char* name) const { return dir_ + "/" + name; }

  // Scores every fixture candidate with the model at `onnx_path`, in
  // expected_*.bin's row layout. A `max_moves` below the candidate count makes
  // the service split the set into chunks.
  std::vector<float> run(const std::string& onnx_path, Precision precision, int max_moves);

  // Engine plans in this test's cache, in no particular order.
  std::vector<std::filesystem::path> cached_plans() const;

  // Worst deviation of `got` from a reference, per field group.
  Tolerance worst_deviation(const std::vector<float>& got,
                            const std::vector<float>& expected) const;

  // Holds the worst deviation to `tol`, printing it under `label`.
  void expect_matches(const std::vector<float>& got, const std::vector<float>& expected,
                      const char* label, Tolerance tol) const;

  std::string dir_;
  std::filesystem::path generated_;   // non-empty iff this test generated the fixture
  std::filesystem::path cache_root_;  // the mount_root the plan cache lives under

  std::vector<float> board_;
  MoveFeatureArrays moves_;
  std::vector<float> expected_a_;
  std::vector<float> expected_b_;
};

void MsetInferenceParityTest::SetUp() {
  cache_root_ = make_cache_root();
  std::filesystem::create_directories(cache_root_);
  if (!g_fixture_dir.empty()) {
    dir_ = g_fixture_dir;
    // ctest's fixture-setup step leaves the directory empty when torch/onnx
    // are unavailable; skip as a self-generating run would.
    if (!std::filesystem::exists(dir_ + "/model_a.onnx")) {
      GTEST_SKIP() << "fixture directory " << dir_ << " is empty; is torch/onnx installed?";
    }
  } else {
    generated_ = make_scratch_dir("scribblez_msetparity_");
    dir_ = generated_.string();
    if (!generate_fixture(dir_)) {
      GTEST_SKIP() << "could not generate fixture; is torch/onnx installed?";
    }
  }

  board_ = read_binary<float>(dir_ + "/board.bin");
  moves_.letters = read_binary<int32_t>(dir_ + "/move_letters.bin");
  moves_.blanks = read_binary<uint8_t>(dir_ + "/move_blanks.bin");
  moves_.squares = read_binary<int32_t>(dir_ + "/move_squares.bin");
  moves_.tile_mask = read_binary<uint8_t>(dir_ + "/move_tile_mask.bin");
  moves_.scalars = read_binary<float>(dir_ + "/move_scalars.bin");
  moves_.count = moves_.scalars.size() / scribblez::move_set::kMoveScalars;
  expected_a_ = read_binary<float>(dir_ + "/expected_a.bin");
  expected_b_ = read_binary<float>(dir_ + "/expected_b.bin");

  ASSERT_GT(moves_.count, 0);
  ASSERT_EQ(expected_a_.size(), size_t(moves_.count) * kFieldsPerRow);
  ASSERT_EQ(expected_b_.size(), expected_a_.size());
}

std::vector<float> MsetInferenceParityTest::run(const std::string& onnx_path, Precision precision,
                                                int max_moves) {
  MsetParams params;
  params.onnx_path = onnx_path;
  params.precision = precision;
  params.max_rows = max_moves;
  params.mount_root = cache_root_.string();
  // Parity checks the inference plumbing, not kernel-tactic quality, so build
  // at optimization level 0 to keep a cold build to a few seconds.
  params.fast_build = true;
  scribblez::nn::TrtEvalService<MoveSetEvaluationSpec> service(params);
  service.load();

  EXPECT_EQ(board_.size(), size_t(service.spatial_planes()) * 225 + size_t(service.scalar_floats()))
    << "the fixture's board row is not the width the loaded model consumes";

  std::vector<float> wld(size_t(moves_.count) * scribblez::nn::WldOutput::kRowElems);
  std::vector<float> sd(size_t(moves_.count) * scribblez::nn::ScoreDiffOutput::kRowElems);
  float* const head_out[] = {wld.data(), sd.data()};
  service.evaluate({board_.data(), &moves_}, head_out);

  std::vector<float> got(size_t(moves_.count) * kFieldsPerRow);
  for (int r = 0; r < moves_.count; ++r) {
    const float* w = wld.data() + size_t(r) * scribblez::nn::WldOutput::kRowElems;
    const float* s = sd.data() + size_t(r) * scribblez::nn::ScoreDiffOutput::kRowElems;
    float* dst = got.data() + size_t(r) * kFieldsPerRow;
    dst[0] = w[0] + 0.5f * w[1];
    dst[1] = w[0];
    dst[2] = w[1];
    dst[3] = w[2];
    dst[4] = s[0];
    dst[5] = s[1];
  }
  return got;
}

std::vector<std::filesystem::path> MsetInferenceParityTest::cached_plans() const {
  std::vector<std::filesystem::path> plans;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(cache_root_)) {
    if (entry.path().extension() == ".engine") plans.push_back(entry.path());
  }
  return plans;
}

Tolerance MsetInferenceParityTest::worst_deviation(const std::vector<float>& got,
                                                   const std::vector<float>& expected) const {
  Tolerance worst{0.0f, 0.0f};
  for (size_t i = 0; i < got.size(); ++i) {
    const int k = i % kFieldsPerRow;
    // Without this check a NaN would vanish: std::max keeps its first argument
    // when the comparison is false, as every comparison with NaN is. An all-NaN
    // run, which is what an unbound device buffer can produce, would then
    // report a perfect match.
    const float deviation = std::abs(got[i] - expected[i]);
    if (!std::isfinite(deviation)) {
      ADD_FAILURE() << "non-finite output: move " << i / kFieldsPerRow << " field " << k << " = "
                    << got[i];
      continue;
    }
    float& tracked = k < 4 ? worst.prob : worst.score_diff;
    tracked = std::max(tracked, deviation);
  }
  return worst;
}

void MsetInferenceParityTest::expect_matches(const std::vector<float>& got,
                                             const std::vector<float>& expected, const char* label,
                                             Tolerance tol) const {
  const Tolerance worst = worst_deviation(got, expected);
  std::cout << "  [" << label << "] max prob err = " << worst.prob << " (tol " << tol.prob
            << "), max score_diff err = " << worst.score_diff << " (tol " << tol.score_diff
            << ")\n";
  EXPECT_LE(worst.prob, tol.prob) << label;
  EXPECT_LE(worst.score_diff, tol.score_diff) << label;
}

// For tests that do not inspect the cache: uses the persistent production plan
// cache under the default mount_root, so later runs skip the TensorRT build.
// This is safe because every cache hit is refitted with the loaded model's own
// weights, and the generator's fixed seed makes the models identical on every
// run. fast_build plans live in their own subtree (engine_plan_cache_path), so
// a test plan can never satisfy a full-optimization production load.
class MsetInferenceParityCachedTest : public MsetInferenceParityTest {
 protected:
  std::filesystem::path make_cache_root() const override {
    return scribblez::nn::NeuralNetParamsBase{}.mount_root;
  }
  bool owns_cache_root() const override { return false; }
};

TEST_F(MsetInferenceParityCachedTest, MatchesPyTorchReferenceAtEveryPrecision) {
  expect_matches(run(model("model_a.onnx"), Precision::kFP32, moves_.count), expected_a_, "FP32",
                 kFp32Tol);
  expect_matches(run(model("model_a.onnx"), Precision::kFP16, moves_.count), expected_a_, "FP16",
                 kFp16Tol);
  expect_matches(run(model("model_a.onnx"), Precision::kBF16, moves_.count), expected_a_, "BF16",
                 kBf16Tol);
}

// A real move set runs to thousands of candidates, so chunking is a normal
// path, not an edge case: every chunk re-sends the board, and the last is
// short.
TEST_F(MsetInferenceParityCachedTest, ChunksACandidateSetLargerThanTheEngine) {
  const int chunk = 8;
  ASSERT_GT(moves_.count, chunk) << "the fixture must exceed the chunk size to test chunking";
  ASSERT_NE(moves_.count % chunk, 0) << "the fixture must leave a short final chunk";

  expect_matches(run(model("model_a.onnx"), Precision::kFP32, chunk), expected_a_, "chunked",
                 kFp32Tol);
}

// The plan cache holds one plan per architecture and refits each checkpoint
// onto it. TensorRT cannot confirm that a refit replaced every weight (see
// NeuralNetBase::Impl::refit_engine), so the outputs are the check: loading A
// twice builds one plan and serves A's reference, and same-architecture B hits
// that plan and serves B's reference, not A's. Needs an empty cache, so it
// uses the scratch one.
TEST_F(MsetInferenceParityTest, SharesOnePlanAcrossCheckpointsAndRefits) {
  expect_matches(run(model("model_a.onnx"), Precision::kFP32, moves_.count), expected_a_, "A cold",
                 kFp32Tol);
  const std::vector<std::filesystem::path> after_cold = cached_plans();
  ASSERT_EQ(after_cold.size(), 1u);
  const std::filesystem::file_time_type built_at = std::filesystem::last_write_time(after_cold[0]);

  expect_matches(run(model("model_a.onnx"), Precision::kFP32, moves_.count), expected_a_,
                 "A cached", kFp32Tol);
  // A rebuild would overwrite the same architecture-keyed file, so only the
  // write time distinguishes reuse from a silent rebuild.
  EXPECT_EQ(std::filesystem::last_write_time(after_cold[0]), built_at)
    << "the same model must reuse its plan, not rebuild it";

  const std::vector<float> from_b = run(model("model_b.onnx"), Precision::kFP32, moves_.count);
  expect_matches(from_b, expected_b_, "B refitted", kFp32Tol);
  EXPECT_EQ(cached_plans().size(), 1u)
    << "a same-architecture checkpoint must refit the shared plan, not build its own";
  EXPECT_EQ(std::filesystem::last_write_time(after_cold[0]), built_at)
    << "a refit must reuse the cached plan, not rebuild it";

  // If the two random-init models happened to agree, the check above would
  // pass even with A's weights.
  EXPECT_GT(worst_deviation(from_b, expected_a_).prob, 100 * kFp32Tol.prob)
    << "the two fixture models are too alike to detect an unrefitted plan";
}

// Models the loader must reject, because nothing downstream would catch them:
//   * a stale move-encoding version has the same shapes and would silently run
//     off distribution;
//   * a move-set model loaded as a position model is a different graph;
//   * narrower move rows or smaller element dtypes build and load fine, but the
//     service stages rows at the encoder's sizes, so feeding them would overrun
//     the engine's buffers.
TEST_F(MsetInferenceParityTest, RejectsModelsThisEncoderMustNotFeed) {
  // Each error is matched against its expected cause, so a load that failed for
  // an unrelated reason (a missing fixture file, say) cannot pass.
  MsetParams stale;
  stale.onnx_path = model("model_stale.onnx");
  stale.mount_root = cache_root_.string();
  stale.fast_build = true;
  const std::string stale_error = load_failure_message(stale);
  EXPECT_NE(stale_error.find("Encoding version mismatch"), std::string::npos) << stale_error;
  EXPECT_NE(stale_error.find("move_encoding_version"), std::string::npos) << stale_error;

  PositionParams position;
  position.onnx_path = model("model_a.onnx");
  position.mount_root = cache_root_.string();
  position.fast_build = true;
  const std::string graph_error = load_failure_message(position);
  EXPECT_NE(graph_error.find("declares graph 'move_set_eval'"), std::string::npos) << graph_error;

  MsetParams narrow;
  narrow.onnx_path = model("model_narrow.onnx");
  narrow.mount_root = cache_root_.string();
  narrow.fast_build = true;
  narrow.max_rows = moves_.count;
  const std::string width_error = load_failure_message(narrow);
  EXPECT_NE(width_error.find("Tensor width mismatch"), std::string::npos) << width_error;

  MsetParams small_elements = narrow;
  small_elements.onnx_path = model("model_uint8_letters.onnx");
  const std::string dtype_error = load_failure_message(small_elements);
  EXPECT_NE(dtype_error.find("Tensor dtype mismatch"), std::string::npos) << dtype_error;
}

}  // namespace

// A custom main so the first non-gtest argument can name a fixture directory.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc >= 2) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
