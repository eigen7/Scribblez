// Suite 1 hop B: confirm the TensorRT inference path (plus the C++ Eval decode)
// reproduces the PyTorch reference the dashboard runs on.
//
// The fixture is a directory holding model.onnx, inputs.bin (N x kInputFloats
// float32), and expected.bin (N x 6 float32: win_prob, p_win, p_draw, p_loss,
// score_diff_mean, score_diff_std, the PyTorch decode), produced by
// py/scripts/gen_nn_parity_fixture.py. This test loads the model through
// NNEvaluationService and evaluates the rows at FP16 -- the precision production
// inference runs -- holding every field to a tight tolerance against the PyTorch
// FP32 reference. FP16 is the coarser precision, so its deviation bounds FP32's;
// passing it validates the engine build, host/device copies, output binding
// order, and the softmax/mean decode. Fails if any field drifts beyond tolerance.
//
// Run it as a one-liner with no arguments:
//   test_nn_inference_parity
// It generates a fresh fixture into a temp dir (by invoking the Python
// generator at the build-time SCRIBBLEZ_PY_DIR), runs the comparison, and
// cleans up. If the fixture cannot be generated (no torch/onnx) it prints a
// skip line and exits 0. Pass an explicit directory to reuse a fixture instead:
//   test_nn_inference_parity <fixture_dir>

#include "scribblez/input_encoder.h"
#include "scribblez/nn/nn_evaluation_service.h"
#include "scribblez/nn/trt_util.h"

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

using scribblez::kInputFloats;
using scribblez::nn::Eval;

// win_prob, p_win, p_draw, p_loss, score_diff_mean, score_diff_std
constexpr int kFieldsPerRow = 6;

// FP16 is compared against the PyTorch FP32 reference at this tight tolerance:
// the tiny fixture model and the points-scale outputs leave little room for
// genuine precision drift, so a real regression (a wrong head, a decode bug)
// shows up well outside these bounds. The test prints the actual max deviations,
// so tune here if a future model legitimately needs more slack.
constexpr float kProbTol = 1e-3f;      // bounds the four probability fields
constexpr float kScoreDiffTol = 0.2f;  // bounds the score-diff mean and std (points)

static std::vector<float> read_floats(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "cannot open " << path << "\n";
    std::exit(2);
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<float> out(static_cast<size_t>(bytes) / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

// Pack an Eval into the same 6-field order as expected.bin.
static void pack(const Eval& e, float* dst) {
  dst[0] = e.win_prob;
  dst[1] = e.p_win;
  dst[2] = e.p_draw;
  dst[3] = e.p_loss;
  dst[4] = e.score_diff_mean;
  dst[5] = e.score_diff_std;
}

// Evaluate every row under `precision` and report the worst per-field deviation
// from the PyTorch reference. Returns false (and prints) if any deviation
// exceeds kProbTol (probability fields) or kScoreDiffTol (score-diff mean).
static bool check_precision(const std::string& onnx_path, scribblez::nn::Precision precision,
                            const char* label, const std::vector<float>& inputs,
                            const std::vector<float>& expected, int n) {
  scribblez::nn::NeuralNetParams params;
  params.onnx_path = onnx_path;
  params.max_batch_size = n;
  params.precision = precision;
  // The parity check validates the inference stack (engine bindings, host/device
  // copies, the C++ decode), not kernel-tactic quality, so build the engine at
  // optimization level 0 to keep the cold engine build to a few seconds.
  params.fast_build = true;
  scribblez::nn::NNEvaluationService service(params);
  service.load();

  std::vector<Eval> evals = service.evaluate(inputs.data(), n);

  float max_prob_err = 0.0f;
  float max_sd_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float got[kFieldsPerRow];
    pack(evals[i], got);
    const float* exp = expected.data() + static_cast<size_t>(i) * kFieldsPerRow;
    for (int k = 0; k < 4; ++k) max_prob_err = std::max(max_prob_err, std::abs(got[k] - exp[k]));
    max_sd_err = std::max(max_sd_err, std::abs(got[4] - exp[4]));  // mean
    max_sd_err = std::max(max_sd_err, std::abs(got[5] - exp[5]));  // std
  }

  const bool ok = max_prob_err <= kProbTol && max_sd_err <= kScoreDiffTol;
  std::cout << "  [" << label << "] max prob err = " << max_prob_err << " (tol " << kProbTol
            << "), max score_diff_mean err = " << max_sd_err << " (tol " << kScoreDiffTol << ") -> "
            << (ok ? "OK" : "FAIL") << "\n";
  return ok;
}

#ifdef SCRIBBLEZ_PY_DIR
// Create a unique scratch directory for a self-generated fixture.
static std::filesystem::path make_scratch_dir() {
  std::filesystem::path base = std::filesystem::temp_directory_path() /
                               ("scribblez_nnparity_" + std::to_string(::time(nullptr)) + "_" +
                                std::to_string(static_cast<unsigned long>(std::random_device{}())));
  std::filesystem::create_directories(base);
  return base;
}

// Invoke the Python fixture generator (at build-time SCRIBBLEZ_PY_DIR) to write
// model.onnx / inputs.bin / expected.bin into `out_dir`. Returns false if the
// generator is unavailable or fails (e.g. torch/onnx not installed).
static bool generate_fixture(const std::string& out_dir) {
  const std::string py_dir = SCRIBBLEZ_PY_DIR;
  const std::string cmd = "cd \"" + py_dir + "\" && PYTHONPATH=\"" + py_dir +
                          "\" python3 -m scripts.gen_nn_parity_fixture --out-dir \"" + out_dir +
                          "\" --num-rows 8";
  return std::system(cmd.c_str()) == 0;
}
#endif

int main(int argc, char** argv) {
  // Use a caller-provided fixture if given; otherwise self-generate one into a
  // temp dir (cleaned up on exit).
  std::string dir;
  std::filesystem::path scratch;  // non-empty iff we created the fixture
  if (argc >= 2) {
    dir = argv[1];
  } else {
#ifdef SCRIBBLEZ_PY_DIR
    scratch = make_scratch_dir();
    dir = scratch.string();
    if (!generate_fixture(dir)) {
      std::cout << "test_nn_inference_parity: SKIPPED (could not generate fixture; "
                   "is torch/onnx installed?)\n";
      std::filesystem::remove_all(scratch);
      return 0;
    }
#else
    std::cout << "test_nn_inference_parity: SKIPPED (built without SCRIBBLEZ_PY_DIR; "
                 "pass a fixture dir as argv[1])\n";
    return 0;
#endif
  }

  const std::string onnx_path = dir + "/model.onnx";
  std::vector<float> inputs = read_floats(dir + "/inputs.bin");
  std::vector<float> expected = read_floats(dir + "/expected.bin");

  bool ok = inputs.size() % kInputFloats == 0;
  if (!ok) {
    std::cerr << "inputs.bin size " << inputs.size() << " not a multiple of kInputFloats "
              << kInputFloats << "\n";
  }
  const int n = ok ? static_cast<int>(inputs.size() / kInputFloats) : 0;
  if (ok && expected.size() != static_cast<size_t>(n) * kFieldsPerRow) {
    std::cerr << "expected.bin has " << expected.size() << " floats; want " << n * kFieldsPerRow
              << " (N=" << n << ")\n";
    ok = false;
  }

  if (ok) {
    std::cout << "test_nn_inference_parity: " << n << " rows from " << dir << "\n";
    ok &= check_precision(onnx_path, scribblez::nn::Precision::kFP16, "FP16", inputs, expected, n);
  }

  if (!scratch.empty()) std::filesystem::remove_all(scratch);

  if (!ok) {
    std::cerr << "test_nn_inference_parity FAILED\n";
    return 1;
  }
  std::cout << "test_nn_inference_parity passed\n";
  return 0;
}
