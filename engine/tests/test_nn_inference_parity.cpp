// Checks that the position-evaluation model served through TensorRT, plus the
// C++ decode of its heads, reproduces the PyTorch FP32 reference. Passing
// validates the engine build, host/device copies, output binding order and the
// decode, at BF16 (the production default) and at FP16.
//
// It does not prove that 16-bit kernels run. TensorRT may serve a 16-bit
// request with FP32 tactics, and on this small fixture it does so for BF16: the
// BF16 run matches the reference to ~1e-5 even at full optimization. So the
// BF16 case is a build/bind/decode smoke test. Reduced-precision accuracy on
// real models is covered in docs/plans/fp16_safe_serving.md.
//
// Fixtures come from py/scripts/position_eval/gen_parity_fixture.py: one
// directory per trunk (conv/, transformer/), each holding
//   model.onnx
//   inputs.bin    N rows x the base input width, float32
//   expected.bin  N rows x 6 float32: win_prob, p_win, p_draw, p_loss,
//                 score_diff_mean, score_diff_std
//
// Run with no arguments, the binary generates fixtures into a temp dir through
// the Python generator at SCRIBBLEZ_PY_DIR, and skips if that fails (no
// torch/onnx). Pass a fixture root as the first non-gtest argument to reuse
// existing fixtures instead; ctest does this so the generator runs once per
// suite rather than once per case.
//   test_nn_inference_parity [<fixture_root>]

#include "encoding/input_encoder.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"

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

// The generator uses the base input layout (no opponent-leave block).
const int kFixtureInputFloats = scribblez::input_floats(scribblez::InputEncodingSpec{nullptr});

// The expected.bin fields, in order (see the file header).
constexpr int kFieldsPerRow = 6;

// Tolerances are sized to each format's mantissa: BF16 has 8 bits to FP16's 10,
// so it gets about 4x the slack. A real regression (a wrong head, a decode bug)
// lands far outside these on the tiny fixture model. The test prints the actual
// maximum deviations, for retuning if a future fixture legitimately needs more.
// The BF16 bounds only bite if a larger fixture makes TensorRT pick real BF16
// kernels.
constexpr float kFp16ProbTol = 1e-3f;      // the four probability fields
constexpr float kFp16ScoreDiffTol = 0.2f;  // score-diff mean and std, in points
constexpr float kBf16ProbTol = 5e-3f;
constexpr float kBf16ScoreDiffTol = 0.8f;

// The fixture root from the command line; empty means self-generate.
static std::string g_fixture_root;

// Must match gen_parity_fixture.py's TRUNK_FIXTURES.
constexpr const char* kTrunks[] = {"conv", "transformer"};

static std::vector<float> read_floats(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    ADD_FAILURE() << "cannot open " << path;
    return {};
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<float> out(size_t(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

// Packs one row's decoded heads in expected.bin's field order.
static void pack(const float* wld, const float* sd, float* dst) {
  dst[0] = wld[0] + 0.5f * wld[1];
  dst[1] = wld[0];
  dst[2] = wld[1];
  dst[3] = wld[2];
  dst[4] = sd[0];
  dst[5] = sd[1];
}

// Evaluates every row at `precision` and bounds the worst per-field deviation
// from the reference.
static void check_precision(const std::string& onnx_path, scribblez::nn::Precision precision,
                            const char* label, float prob_tol, float sd_tol,
                            const std::vector<float>& inputs, const std::vector<float>& expected,
                            int n) {
  using Spec = scribblez::nn::PositionEvaluationSpec;
  scribblez::nn::NeuralNetParams<Spec> params;
  params.onnx_path = onnx_path;
  params.max_rows = n;
  params.precision = precision;
  // Parity checks the inference plumbing, not kernel-tactic quality, so build
  // at optimization level 0 to keep the cold engine build to a few seconds.
  params.fast_build = true;
  scribblez::nn::TrtEvalService<Spec> service(params);
  service.load();

  std::vector<float> wld(size_t(n) * scribblez::nn::WldOutput::kRowElems);
  std::vector<float> sd(size_t(n) * scribblez::nn::ScoreDiffOutput::kRowElems);
  float* const head_out[] = {wld.data(), sd.data()};
  service.evaluate({inputs.data(), n}, head_out);

  float max_prob_err = 0.0f;
  float max_sd_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float got[kFieldsPerRow];
    pack(wld.data() + size_t(i) * scribblez::nn::WldOutput::kRowElems,
         sd.data() + size_t(i) * scribblez::nn::ScoreDiffOutput::kRowElems, got);
    const float* exp = expected.data() + size_t(i) * kFieldsPerRow;
    for (int k = 0; k < 4; ++k) max_prob_err = std::max(max_prob_err, std::abs(got[k] - exp[k]));
    max_sd_err = std::max(max_sd_err, std::abs(got[4] - exp[4]));  // mean
    max_sd_err = std::max(max_sd_err, std::abs(got[5] - exp[5]));  // std
  }

  std::cout << "  [" << label << "] max prob err = " << max_prob_err << " (tol " << prob_tol
            << "), max score_diff_mean err = " << max_sd_err << " (tol " << sd_tol << ")\n";
  EXPECT_LE(max_prob_err, prob_tol);
  EXPECT_LE(max_sd_err, sd_tol);
}

#ifdef SCRIBBLEZ_PY_DIR
static std::filesystem::path make_scratch_dir() {
  std::filesystem::path base = std::filesystem::temp_directory_path() /
                               ("scribblez_nnparity_" + std::to_string(::time(nullptr)) + "_" +
                                std::to_string(std::random_device{}()));
  std::filesystem::create_directories(base);
  return base;
}

// Writes every trunk's fixture under `out_dir`; false if the generator fails
// (for example, torch or onnx is not installed).
static bool generate_fixtures(const std::string& out_dir) {
  const std::string py_dir = SCRIBBLEZ_PY_DIR;
  const std::string cmd = "cd \"" + py_dir + "\" && PYTHONPATH=\"" + py_dir +
                          "\" python3 -m scripts.position_eval.gen_parity_fixture --out-dir \"" +
                          out_dir + "\" --num-rows 8";
  return std::system(cmd.c_str()) == 0;
}
#endif

// Parameterised over the trunk. Each case uses the command-line fixture root
// if one was given, else generates its own into a scratch dir it removes.
class NnInferenceParityTest : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!scratch_.empty()) std::filesystem::remove_all(scratch_);
  }

  std::string dir_;
  std::filesystem::path scratch_;  // non-empty iff this test created the fixtures
};

void NnInferenceParityTest::SetUp() {
  if (!g_fixture_root.empty()) {
    dir_ = g_fixture_root + "/" + GetParam();
    // ctest's fixture-setup step leaves the directory empty when torch/onnx
    // are unavailable; skip as a self-generating run would.
    if (!std::filesystem::exists(dir_ + "/model.onnx")) {
      GTEST_SKIP() << "fixture directory " << dir_ << " is empty; is torch/onnx installed?";
    }
    return;
  }
#ifdef SCRIBBLEZ_PY_DIR
  scratch_ = make_scratch_dir();
  dir_ = (scratch_ / GetParam()).string();
  if (!generate_fixtures(scratch_.string())) {
    GTEST_SKIP() << "could not generate fixtures; is torch/onnx installed?";
  }
#else
  GTEST_SKIP() << "built without SCRIBBLEZ_PY_DIR; pass a fixture dir on the command line";
#endif
}

// Loads and shape-checks the fixture rows; returns the row count.
static int load_fixture(const std::string& dir, std::vector<float>* inputs,
                        std::vector<float>* expected) {
  *inputs = read_floats(dir + "/inputs.bin");
  *expected = read_floats(dir + "/expected.bin");
  EXPECT_EQ(inputs->size() % kFixtureInputFloats, 0u)
    << "inputs.bin size " << inputs->size() << " not a multiple of the full input width "
    << kFixtureInputFloats;
  const int n = inputs->size() / kFixtureInputFloats;
  EXPECT_GT(n, 0);
  EXPECT_EQ(expected->size(), size_t(n) * kFieldsPerRow) << "(N=" << n << ")";
  return n;
}

TEST_P(NnInferenceParityTest, Bf16MatchesPyTorchReference) {
  std::vector<float> inputs, expected;
  const int n = load_fixture(dir_, &inputs, &expected);
  ASSERT_GT(n, 0);
  std::cout << "  " << n << " rows from " << dir_ << "\n";
  check_precision(dir_ + "/model.onnx", scribblez::nn::Precision::kBF16, "BF16", kBf16ProbTol,
                  kBf16ScoreDiffTol, inputs, expected, n);
}

TEST_P(NnInferenceParityTest, Fp16MatchesPyTorchReference) {
  std::vector<float> inputs, expected;
  const int n = load_fixture(dir_, &inputs, &expected);
  ASSERT_GT(n, 0);
  std::cout << "  " << n << " rows from " << dir_ << "\n";
  check_precision(dir_ + "/model.onnx", scribblez::nn::Precision::kFP16, "FP16", kFp16ProbTol,
                  kFp16ScoreDiffTol, inputs, expected, n);
}

// PositionEvalService::create() returns one shared instance for equal params,
// so a run's threads share one loaded engine and its execution-context memory,
// and a distinct instance when an engine-determining field differs. It lives in
// this suite because create() needs a real model to build.
TEST_P(NnInferenceParityTest, CreateSharesOneServicePerParams) {
  scribblez::nn::NeuralNetParams<scribblez::nn::PositionEvaluationSpec> params;
  params.onnx_path = dir_ + "/model.onnx";
  params.precision = scribblez::nn::Precision::kBF16;

  std::shared_ptr<scribblez::nn::PositionEvalService> a =
    scribblez::nn::PositionEvalService::create(params);
  std::shared_ptr<scribblez::nn::PositionEvalService> b =
    scribblez::nn::PositionEvalService::create(params);
  EXPECT_EQ(a.get(), b.get()) << "equal params must share one loaded service";

  scribblez::nn::NeuralNetParams<scribblez::nn::PositionEvaluationSpec> other = params;
  other.max_rows = params.max_rows + 1;  // engine-determining
  std::shared_ptr<scribblez::nn::PositionEvalService> c =
    scribblez::nn::PositionEvalService::create(other);
  EXPECT_NE(a.get(), c.get()) << "differing params must not share";
}

INSTANTIATE_TEST_SUITE_P(Trunks, NnInferenceParityTest, ::testing::ValuesIn(kTrunks),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                           return std::string(info.param);
                         });

// A custom main so the first non-gtest argument can name a fixture root.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc >= 2) g_fixture_root = argv[1];
  return RUN_ALL_TESTS();
}
