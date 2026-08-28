// The move-proposal two-graph runtime (agent/move_proposal_runtime.h) against
// the PyTorch model it serves (roadmap item 3). mset_infer_parity's counterpart
// for the split evidence-path graphs.
//
// The runtime runs the cache graph once, then -- for each evidence set the
// fixture describes -- stages the raw sim observations through
// agent/evidence_staging.h and runs the step graph, and this test compares
// every candidate's conditioned prediction against MoveSetEvalModel.forward's.
// More can go silently wrong here than in the single-graph runtimes: the
// board/g/move_enc host handoff, the leading-1 evidence inputs, the move_enc
// gather by scored index, and the empty-set fusion gate. Each changes the
// numbers, and every case is a comparison against the PyTorch reference for the
// same inputs. The verification is tolerance-bounded, not bit-identical:
// independent TensorRT plans reorder float sums, so parity is held to a
// tolerance (docs/roadmap.md item 3), as the sibling mset parity test is.
//
// The fixture (one model's cache/step pair, a candidate set, its raw Move /
// SimObservation records, and several evidence cases -- empty, partial with
// scattered+duplicate indices, and full at the padding boundary) is generated
// by py/scripts/move_set_eval/gen_proposal_parity_fixture.py, invoked at the
// build-time SCRIBBLEZ_PY_DIR; the test skips when that generator cannot run
// (no torch/onnx). Pass a fixture directory as the first non-gtest argument to
// reuse one:
//   test_proposal_inference_parity <fixture_dir>

#include "agent/move_proposal_runtime.h"
#include "encoding/input_encoder.h"
#include "game/move.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "training/move_set_encoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

using scribblez::Move;
using scribblez::SimObservation;
using scribblez::agent::MoveProposalRuntime;

namespace {

// The fixture's per-candidate reference scalars: [p_win, p_draw, p_loss,
// sd_mean, sd_std, gain]. The plane reference is a separate M x 4 x 225 file.
constexpr int kScalarFields = 6;
constexpr int kPlaneFloats = scribblez::nn::kNumPlacementPlanes * scribblez::kBoardCells;

// How far a TensorRT run may deviate from the PyTorch FP32 reference. Set from
// the measured noise floor, not loosely: on this fixture the observed worst
// deviations are ~1e-5 (wld probs), ~3.5e-4 (planes, a 900-cell sigmoid off a
// 225-token attention -- the jitteriest head), and ~5e-5 (score_diff / gain).
// The bounds sit above those with room for cross-GPU / cross-build kernel jitter
// -- ~6x on the jittery planes head, a wider (~50-100x) round-number margin on
// the far tighter wld / score_diff / gain -- yet stay orders of magnitude below
// any real defect (a dropped ev_obs field, a mis-strided evidence plane, a
// mis-bound handoff move outputs by far more). The test prints the actual
// deviations, so retune here if a future model legitimately needs it.
struct Tolerance {
  float prob;        // the three WLD probabilities
  float planes;      // the sigmoid placement planes (widest head)
  float score_diff;  // the score-diff mean/std, in points
  float gain;        // the proves-best gain, in points
};
constexpr Tolerance kFp32Tol{5e-4f, 2e-3f, 5e-3f, 5e-3f};

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
  if (bytes) f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

std::filesystem::path make_scratch_dir(const char* prefix) {
  std::filesystem::path base =
    std::filesystem::temp_directory_path() /
    (prefix + std::to_string(::time(nullptr)) + "_" + std::to_string(std::random_device{}()));
  std::filesystem::create_directories(base);
  return base;
}

bool generate_fixture(const std::string& out_dir) {
#ifdef SCRIBBLEZ_PY_DIR
  const std::string py_dir = SCRIBBLEZ_PY_DIR;
  const std::string cmd = "cd \"" + py_dir + "\" && PYTHONPATH=\"" + py_dir +
                          "\" python3 -m scripts.move_set_eval.gen_proposal_parity_fixture "
                          "--out-dir \"" +
                          out_dir + "\"";
  return std::system(cmd.c_str()) == 0;
#else
  (void)out_dir;
  return false;
#endif
}

// One evidence case as the fixture describes it.
struct EvidenceCase {
  std::string name;
  std::vector<int> indices;        // scored indices, empty for the empty case
  std::vector<float> ref_scalars;  // M x kScalarFields
  std::vector<float> ref_planes;   // M x kPlaneFloats
};

// The worst deviation of a conditioned prediction from the reference, per field
// group. NaN is caught rather than hidden (std::max returns its first argument
// on a NaN comparison, so an all-NaN run -- what an unbound buffer looks like --
// would otherwise report a perfect match).
struct Worst {
  float prob = 0, score_diff = 0, gain = 0, planes = 0;
};

void track(float& worst, float got, float want, const char* what, int move) {
  const float dev = std::abs(got - want);
  if (!std::isfinite(dev)) {
    ADD_FAILURE() << "non-finite output: " << what << " move " << move << " = " << got;
    return;
  }
  worst = std::max(worst, dev);
}

class ProposalInferenceParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    if (!cache_root_.empty()) std::filesystem::remove_all(cache_root_);
  }

  std::string model(const char* name) const { return dir_ + "/" + name; }

  // A loaded runtime over the fixture's graphs, its plan cache in this test's
  // scratch root, engines bounded at `max_rows` (below the candidate count to
  // exercise chunking of both graphs).
  MoveProposalRuntime make_runtime(int max_rows) const;

  std::string dir_;
  std::filesystem::path generated_;
  std::filesystem::path cache_root_;

  int num_moves_ = 0;
  std::vector<float> board_;
  scribblez::move_set::MoveFeatureArrays moves_;
  std::vector<Move> sobs_moves_;
  std::vector<SimObservation> obs_;
  std::vector<EvidenceCase> cases_;
};

void ProposalInferenceParityTest::SetUp() {
  cache_root_ = make_scratch_dir("scribblez_propcache_");
  if (!g_fixture_dir.empty()) {
    dir_ = g_fixture_dir;
    if (!std::filesystem::exists(dir_ + "/cache.onnx")) {
      GTEST_SKIP() << "fixture directory " << dir_ << " is empty; is torch/onnx installed?";
    }
  } else {
    generated_ = make_scratch_dir("scribblez_propparity_");
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
  moves_.count = int(moves_.scalars.size() / scribblez::move_set::kMoveScalars);
  num_moves_ = moves_.count;
  sobs_moves_ = read_binary<Move>(dir_ + "/moves_sobs.bin");
  obs_ = read_binary<SimObservation>(dir_ + "/obs.bin");

  ASSERT_GT(num_moves_, 0);
  ASSERT_EQ(int(sobs_moves_.size()), num_moves_);
  ASSERT_EQ(int(obs_.size()), num_moves_);

  std::ifstream cases(dir_ + "/cases.txt");
  ASSERT_TRUE(cases) << "missing cases.txt";
  std::string name;
  int k = 0;
  while (cases >> name >> k) {
    EvidenceCase c;
    c.name = name;
    c.indices = read_binary<int>(dir_ + "/case_" + name + "_indices.bin");
    c.ref_scalars = read_binary<float>(dir_ + "/case_" + name + "_scalars.bin");
    c.ref_planes = read_binary<float>(dir_ + "/case_" + name + "_planes.bin");
    ASSERT_EQ(int(c.indices.size()), k) << name;
    ASSERT_EQ(c.ref_scalars.size(), size_t(num_moves_) * kScalarFields) << name;
    ASSERT_EQ(c.ref_planes.size(), size_t(num_moves_) * kPlaneFloats) << name;
    cases_.push_back(std::move(c));
  }
  ASSERT_FALSE(cases_.empty());
}

MoveProposalRuntime ProposalInferenceParityTest::make_runtime(int max_rows) const {
  MoveProposalRuntime::Params params;
  params.cache_onnx_path = model("cache.onnx");
  params.step_onnx_path = model("step.onnx");
  params.precision = scribblez::nn::Precision::kFP32;  // item-3 serving precision
  params.max_rows = max_rows;
  params.mount_root = cache_root_.string();
  // The parity check validates the inference stack, not kernel-tactic quality,
  // so build at optimization level 0 to keep cold builds to a few seconds.
  params.fast_build = true;
  return MoveProposalRuntime(params);
}

// Every candidate's conditioned prediction against the reference, for one case.
void expect_case_matches(const scribblez::agent::MoveProposalPredictions& got,
                         const EvidenceCase& c, int num_moves, Tolerance tol) {
  Worst worst;
  for (int m = 0; m < num_moves; ++m) {
    const float* ref = c.ref_scalars.data() + size_t(m) * kScalarFields;
    const float* w = got.wld.data() + size_t(m) * 3;
    for (int i = 0; i < 3; ++i) track(worst.prob, w[i], ref[i], "wld", m);
    const float* sd = got.score_diff.data() + size_t(m) * 2;
    track(worst.score_diff, sd[0], ref[3], "sd_mean", m);
    track(worst.score_diff, sd[1], ref[4], "sd_std", m);
    track(worst.gain, got.gain[m], ref[5], "gain", m);

    const float* ref_pl = c.ref_planes.data() + size_t(m) * kPlaneFloats;
    const float* pl = got.planes.data() + size_t(m) * kPlaneFloats;
    for (int i = 0; i < kPlaneFloats; ++i) track(worst.planes, pl[i], ref_pl[i], "planes", m);
  }
  std::cout << "  [" << c.name << "] max prob err = " << worst.prob << " (tol " << tol.prob
            << "), planes " << worst.planes << " (tol " << tol.planes << "), score_diff "
            << worst.score_diff << " (tol " << tol.score_diff << "), gain " << worst.gain
            << " (tol " << tol.gain << ")\n";
  EXPECT_LE(worst.prob, tol.prob) << c.name;
  EXPECT_LE(worst.planes, tol.planes) << c.name;
  EXPECT_LE(worst.score_diff, tol.score_diff) << c.name;
  EXPECT_LE(worst.gain, tol.gain) << c.name;
}

}  // namespace

// The full run at a single chunk: every candidate scored in one predict() per
// graph.
TEST_F(ProposalInferenceParityTest, MatchesPyTorchReferenceForEveryEvidenceCase) {
  MoveProposalRuntime runtime = make_runtime(num_moves_);
  runtime.load();
  const auto plain = runtime.encode(board_.data(), moves_);
  const std::vector<float> plain_wld = plain.wld;
  const std::vector<float> plain_planes = plain.planes;
  const std::vector<float> plain_sd = plain.score_diff;

  for (const EvidenceCase& c : cases_) {
    std::vector<Move> ev_moves;
    std::vector<SimObservation> ev_obs;
    for (int idx : c.indices) {
      ev_moves.push_back(sobs_moves_[idx]);
      ev_obs.push_back(obs_[idx]);
    }
    const auto& conditioned = runtime.condition(ev_moves, ev_obs, std::span<const int>(c.indices));
    expect_case_matches(conditioned, c, num_moves_, kFp32Tol);

    if (c.name == "empty") {
      // The step graph at an empty set must reproduce the cache's plain heads
      // (and the plain mset outputs, which the reference IS), within tolerance.
      Worst worst;
      for (size_t i = 0; i < plain_wld.size(); ++i)
        track(worst.prob, conditioned.wld[i], plain_wld[i], "empty==plain wld", int(i / 3));
      for (size_t i = 0; i < plain_sd.size(); ++i)
        track(worst.score_diff, conditioned.score_diff[i], plain_sd[i], "empty==plain sd",
              int(i / 2));
      for (size_t i = 0; i < plain_planes.size(); ++i)
        track(worst.planes, conditioned.planes[i], plain_planes[i], "empty==plain planes",
              int(i / kPlaneFloats));
      std::cout << "  [empty==plain] wld " << worst.prob << ", planes " << worst.planes << ", sd "
                << worst.score_diff << "\n";
      EXPECT_LE(worst.prob, kFp32Tol.prob);
      EXPECT_LE(worst.planes, kFp32Tol.planes);
      EXPECT_LE(worst.score_diff, kFp32Tol.score_diff);
    }
  }
}

// The same candidate set scored with the engines bounded below M, so both the
// cache and step graphs run in chunks and the runtime's full-M retention of
// move_enc / board / g across chunk boundaries is exercised.
TEST_F(ProposalInferenceParityTest, ChunksACandidateSetLargerThanTheEngines) {
  const int chunk = 32;
  ASSERT_GT(num_moves_, chunk) << "the fixture must exceed the chunk size to test chunking";
  ASSERT_NE(num_moves_ % chunk, 0) << "the fixture must leave a short final chunk";

  MoveProposalRuntime runtime = make_runtime(chunk);
  runtime.load();
  runtime.encode(board_.data(), moves_);
  for (const EvidenceCase& c : cases_) {
    std::vector<Move> ev_moves;
    std::vector<SimObservation> ev_obs;
    for (int idx : c.indices) {
      ev_moves.push_back(sobs_moves_[idx]);
      ev_obs.push_back(obs_[idx]);
    }
    const auto& conditioned = runtime.condition(ev_moves, ev_obs, std::span<const int>(c.indices));
    expect_case_matches(conditioned, c, num_moves_, kFp32Tol);
  }
}

// A cache and step graph exported from different checkpoints (same architecture,
// so C and the layout checks agree, but different weights) share no
// proposal_export_id -- load() must reject the pair loudly rather than serve
// plausible-looking wrong numbers. step_mismatch.onnx is that second model's
// step graph; the mismatch is caught only by the fingerprint, so this is the one
// test of that guard.
TEST_F(ProposalInferenceParityTest, RejectsACacheStepPairFromDifferentModels) {
  MoveProposalRuntime::Params params;
  params.cache_onnx_path = model("cache.onnx");
  params.step_onnx_path = model("step_mismatch.onnx");
  params.precision = scribblez::nn::Precision::kFP32;
  params.max_rows = num_moves_;
  params.mount_root = cache_root_.string();
  params.fast_build = true;
  MoveProposalRuntime runtime(params);
  try {
    runtime.load();
    ADD_FAILURE() << "loading a cache/step pair from different models should have been rejected";
  } catch (const std::runtime_error& e) {
    // Matched against what it should object to, so a load that failed for some
    // unrelated reason cannot pass for the guard doing its job.
    const std::string msg = e.what();
    EXPECT_NE(msg.find("different models"), std::string::npos) << msg;
    EXPECT_NE(msg.find("proposal_export_id"), std::string::npos) << msg;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc >= 2) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
