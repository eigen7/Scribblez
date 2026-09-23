// The move proposal model's two-graph TensorRT runtime
// (agent/move_proposal_nets.h, agent/move_proposal_session.h), checked against
// the PyTorch reference. The test_mset_inference_parity counterpart for the
// evidence path.
//
// A session runs the cache graph once per position, then the step graph once
// per evidence set, after staging the raw sim observations through
// agent/evidence_staging.h. More can go silently wrong than in the single-graph
// runtimes: the board/g/move_enc handoff from cache to step through host
// memory, the leading-1 evidence inputs, the move_enc gather by scored index,
// and the empty-set fusion gate. The cache graph's placement planes, which the
// step graph does not re-emit, are checked on the session's retained cache.
//
// Parity is tolerance-bounded, not bitwise: independent TensorRT plans reorder
// float sums.
//
// The fixture comes from py/scripts/move_set_eval/gen_proposal_parity_fixture.py:
// one model's cache/step pair, a candidate set with its raw Move and
// SimObservation records, and evidence cases (empty; partial, with scattered
// and duplicate indices; full, at the padding boundary). Run with no
// arguments, the binary generates it through the Python generator at
// SCRIBBLEZ_PY_DIR and skips if that fails (no torch/onnx). Pass a fixture
// directory as the first non-gtest argument to reuse one instead:
//   test_proposal_inference_parity [<fixture_dir>]

#include "agent/move_proposal_nets.h"
#include "agent/move_proposal_session.h"
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
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using scribblez::Move;
using scribblez::SimObservation;
using scribblez::agent::EvidenceSet;
using scribblez::agent::MoveProposalNets;
using scribblez::agent::MoveProposalPredictions;
using scribblez::agent::MoveProposalSession;

namespace {

// Per-candidate reference scalars: p_win, p_draw, p_loss, sd_mean, sd_std,
// gain. The plane reference is a separate file of M rows of kPlaneFloats.
constexpr int kScalarFields = 6;
constexpr int kPlaneFloats = scribblez::nn::PlanesOutput::kRowElems;

// Allowed deviation from the PyTorch FP32 reference, set from the measured
// noise floor. The observed worst deviations on this fixture are ~1e-5 for the
// WLD probabilities, ~5e-5 for score_diff and gain, and ~3.5e-4 for the planes,
// the noisiest head (a softmax over 900 cells, then per-cell marginals, on top
// of a 225-token attention). The bounds leave room for kernel differences
// across GPUs and builds (~6x on the planes, a rounder 50-100x elsewhere) yet
// stay orders of magnitude below a real defect such as a dropped evidence
// field, a mis-strided evidence plane or a mis-bound handoff. The test prints
// the actual deviations, for retuning if a future model legitimately needs it.
struct Tolerance {
  float prob;        // the three WLD probabilities
  float planes;      // the per-cell-marginal placement planes (widest head)
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

struct EvidenceCase {
  std::string name;
  std::vector<int> indices;        // scored indices, empty for the empty case
  std::vector<float> ref_scalars;  // M x kScalarFields
};

// The worst deviation of a prediction from the reference, per field group.
struct Worst {
  float prob = 0, score_diff = 0, gain = 0, planes = 0;
};

// Folds one deviation into `worst`, failing on a non-finite one. std::max would
// silently drop a NaN (it keeps its first argument when the comparison is
// false), so an all-NaN run, which an unbound buffer can produce, would report
// a perfect match.
void track(float& worst, float got, float want, const char* what, int move) {
  const float dev = std::abs(got - want);
  if (!std::isfinite(dev)) {
    ADD_FAILURE() << "non-finite output: " << what << " move " << move << " = " << got;
    return;
  }
  worst = std::max(worst, dev);
}

// The first `count` candidates of `moves`, as their own set.
scribblez::move_set::MoveFeatureArrays truncate_moves(
  const scribblez::move_set::MoveFeatureArrays& moves, int count) {
  using namespace scribblez::move_set;
  MoveFeatureArrays out;
  out.count = count;
  out.letters.assign(moves.letters.begin(), moves.letters.begin() + count * kMoveMaxPlaced);
  out.blanks.assign(moves.blanks.begin(), moves.blanks.begin() + count * kMoveMaxPlaced);
  out.squares.assign(moves.squares.begin(), moves.squares.begin() + count * kMoveMaxPlaced);
  out.tile_mask.assign(moves.tile_mask.begin(), moves.tile_mask.begin() + count * kMoveMaxPlaced);
  out.scalars.assign(moves.scalars.begin(), moves.scalars.begin() + count * kMoveScalars);
  return out;
}

class ProposalInferenceParityTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override {
    if (!generated_.empty()) std::filesystem::remove_all(generated_);
    if (!cache_root_.empty()) std::filesystem::remove_all(cache_root_);
  }

  std::string model(const char* name) const { return dir_ + "/" + name; }

  // Params for the fixture's graph pair, with the plan cache in this test's
  // scratch root. A `max_rows` below the candidate count makes both graphs run
  // in chunks.
  MoveProposalNets::Params nets_params(int max_rows) const;

  // The evidence set of one case: the fixture's raw records at its indices.
  EvidenceSet evidence_of(const EvidenceCase& c) const;

  std::string dir_;
  std::filesystem::path generated_;
  std::filesystem::path cache_root_;

  int num_moves_ = 0;
  std::vector<float> board_;
  scribblez::move_set::MoveFeatureArrays moves_;
  std::vector<Move> sobs_moves_;
  std::vector<SimObservation> obs_;
  std::vector<float> plain_planes_;  // M x kPlaneFloats
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
  plain_planes_ = read_binary<float>(dir_ + "/plain_planes.bin");

  ASSERT_GT(num_moves_, 0);
  ASSERT_EQ(int(sobs_moves_.size()), num_moves_);
  ASSERT_EQ(int(obs_.size()), num_moves_);
  ASSERT_EQ(plain_planes_.size(), size_t(num_moves_) * kPlaneFloats);

  std::ifstream cases(dir_ + "/cases.txt");
  ASSERT_TRUE(cases) << "missing cases.txt";
  std::string name;
  int k = 0;
  while (cases >> name >> k) {
    EvidenceCase c;
    c.name = name;
    c.indices = read_binary<int>(dir_ + "/case_" + name + "_indices.bin");
    c.ref_scalars = read_binary<float>(dir_ + "/case_" + name + "_scalars.bin");
    ASSERT_EQ(int(c.indices.size()), k) << name;
    ASSERT_EQ(c.ref_scalars.size(), size_t(num_moves_) * kScalarFields) << name;
    cases_.push_back(std::move(c));
  }
  ASSERT_FALSE(cases_.empty());
}

MoveProposalNets::Params ProposalInferenceParityTest::nets_params(int max_rows) const {
  MoveProposalNets::Params params;
  params.cache_onnx_path = model("cache.onnx");
  params.step_onnx_path = model("step.onnx");
  params.precision = scribblez::nn::Precision::kFP32;  // the serving precision
  params.max_rows = max_rows;
  params.step_max_rows = max_rows;
  params.mount_root = cache_root_.string();
  // Parity checks the inference plumbing, not kernel-tactic quality, so build
  // at optimization level 0 to keep cold builds to a few seconds.
  params.fast_build = true;
  return params;
}

EvidenceSet ProposalInferenceParityTest::evidence_of(const EvidenceCase& c) const {
  EvidenceSet evidence;
  for (int idx : c.indices) evidence.add(sobs_moves_[idx], obs_[idx], idx);
  return evidence;
}

void expect_case_matches(const MoveProposalPredictions& got, const EvidenceCase& c, int num_moves,
                         Tolerance tol) {
  Worst worst;
  for (int m = 0; m < num_moves; ++m) {
    const float* ref = c.ref_scalars.data() + size_t(m) * kScalarFields;
    const float* w = got.wld.data() + size_t(m) * 3;
    for (int i = 0; i < 3; ++i) track(worst.prob, w[i], ref[i], "wld", m);
    const float* sd = got.score_diff.data() + size_t(m) * 2;
    track(worst.score_diff, sd[0], ref[3], "sd_mean", m);
    track(worst.score_diff, sd[1], ref[4], "sd_std", m);
    track(worst.gain, got.gain[m], ref[5], "gain", m);
  }
  std::cout << "  [" << c.name << "] max prob err = " << worst.prob << " (tol " << tol.prob
            << "), score_diff " << worst.score_diff << " (tol " << tol.score_diff << "), gain "
            << worst.gain << " (tol " << tol.gain << ")\n";
  EXPECT_LE(worst.prob, tol.prob) << c.name;
  EXPECT_LE(worst.score_diff, tol.score_diff) << c.name;
  EXPECT_LE(worst.gain, tol.gain) << c.name;
}

void expect_planes_match(const std::vector<float>& got, const std::vector<float>& ref,
                         int num_moves, Tolerance tol) {
  ASSERT_EQ(got.size(), size_t(num_moves) * kPlaneFloats);
  Worst worst;
  for (size_t i = 0; i < got.size(); ++i)
    track(worst.planes, got[i], ref[i], "planes", int(i / kPlaneFloats));
  std::cout << "  [cache planes] max err = " << worst.planes << " (tol " << tol.planes << ")\n";
  EXPECT_LE(worst.planes, tol.planes);
}

// Two predictions over the same candidates, held to `tol` on every field.
void expect_same_predictions(const MoveProposalPredictions& a, const MoveProposalPredictions& b,
                             Tolerance tol, const char* what) {
  ASSERT_EQ(a.num_moves, b.num_moves) << what;
  Worst worst;
  for (size_t i = 0; i < a.wld.size(); ++i) track(worst.prob, a.wld[i], b.wld[i], what, int(i / 3));
  for (size_t i = 0; i < a.score_diff.size(); ++i)
    track(worst.score_diff, a.score_diff[i], b.score_diff[i], what, int(i / 2));
  for (size_t i = 0; i < a.gain.size(); ++i) track(worst.gain, a.gain[i], b.gain[i], what, int(i));
  EXPECT_LE(worst.prob, tol.prob) << what;
  EXPECT_LE(worst.score_diff, tol.score_diff) << what;
  EXPECT_LE(worst.gain, tol.gain) << what;
}

}  // namespace

// Every candidate scored in a single chunk per graph.
TEST_F(ProposalInferenceParityTest, MatchesPyTorchReferenceForEveryEvidenceCase) {
  MoveProposalSession session(MoveProposalNets::create(nets_params(num_moves_)));
  // The fixture stamps its pair as trained at the full padded evidence width.
  EXPECT_EQ(session.nets().trained_max_evidence(), scribblez::nn::kMaxEvidence);
  const MoveProposalPredictions plain = session.encode(board_.data(), moves_);
  ASSERT_TRUE(plain.gain.empty()) << "the cache graph emits no gain head";
  expect_planes_match(session.cache().planes, plain_planes_, num_moves_, kFp32Tol);

  for (const EvidenceCase& c : cases_) {
    const MoveProposalPredictions& conditioned = session.condition(evidence_of(c));
    expect_case_matches(conditioned, c, num_moves_, kFp32Tol);

    if (c.name == "empty") {
      // With no evidence the step graph must reproduce the cache graph's heads.
      Worst worst;
      for (size_t i = 0; i < plain.wld.size(); ++i)
        track(worst.prob, conditioned.wld[i], plain.wld[i], "empty==plain wld", int(i / 3));
      for (size_t i = 0; i < plain.score_diff.size(); ++i)
        track(worst.score_diff, conditioned.score_diff[i], plain.score_diff[i], "empty==plain sd",
              int(i / 2));
      std::cout << "  [empty==plain] wld " << worst.prob << ", sd " << worst.score_diff << "\n";
      EXPECT_LE(worst.prob, kFp32Tol.prob);
      EXPECT_LE(worst.score_diff, kFp32Tol.score_diff);
    }
  }
}

// Engines bounded below M make both graphs run in chunks, which exercises the
// session's retention of the full-M board/g/move_enc handoff across chunk
// boundaries.
TEST_F(ProposalInferenceParityTest, ChunksACandidateSetLargerThanTheEngines) {
  const int chunk = 32;
  ASSERT_GT(num_moves_, chunk) << "the fixture must exceed the chunk size to test chunking";
  ASSERT_NE(num_moves_ % chunk, 0) << "the fixture must leave a short final chunk";

  MoveProposalSession session(MoveProposalNets::create(nets_params(chunk)));
  session.encode(board_.data(), moves_);
  expect_planes_match(session.cache().planes, plain_planes_, num_moves_, kFp32Tol);
  for (const EvidenceCase& c : cases_) {
    expect_case_matches(session.condition(evidence_of(c)), c, num_moves_, kFp32Tol);
  }
}

// One consumer of a shared pair: a candidate subset, the evidence within it,
// and what the consumer produces when it has the nets to itself.
struct ConsumerCase {
  scribblez::move_set::MoveFeatureArrays moves;
  EvidenceSet evidence;
  MoveProposalPredictions want_plain;
  MoveProposalPredictions want_conditioned;
};

// One consumer's worst deviation from its solo reference over rounds of
// encode, condition(empty), condition. Runs on its own thread, so it records
// rather than asserts.
struct ConsumerRun {
  const ConsumerCase* c;
  Worst worst;
  void run(std::shared_ptr<MoveProposalNets> nets, const float* board, int iterations);
};

void ConsumerRun::run(std::shared_ptr<MoveProposalNets> nets, const float* board, int iterations) {
  MoveProposalSession session(std::move(nets));
  for (int i = 0; i < iterations; ++i) {
    const MoveProposalPredictions& plain = session.encode(board, c->moves);
    for (size_t k = 0; k < plain.wld.size(); ++k)
      worst.prob = std::max(worst.prob, std::abs(plain.wld[k] - c->want_plain.wld[k]));
    session.condition(EvidenceSet{});
    const MoveProposalPredictions& got = session.condition(c->evidence);
    for (size_t k = 0; k < got.wld.size(); ++k)
      worst.prob = std::max(worst.prob, std::abs(got.wld[k] - c->want_conditioned.wld[k]));
    for (size_t k = 0; k < got.score_diff.size(); ++k)
      worst.score_diff =
        std::max(worst.score_diff, std::abs(got.score_diff[k] - c->want_conditioned.score_diff[k]));
    for (size_t k = 0; k < got.gain.size(); ++k)
      worst.gain = std::max(worst.gain, std::abs(got.gain[k] - c->want_conditioned.gain[k]));
  }
}

// Sessions sharing one pair each reproduce their solo outputs when their calls
// interleave: the nets hold no per-position state, and the lock spans a whole
// call. Checked first interleaved on one thread (sequencing), then from
// concurrent threads (locking). The failure this guards against is one session
// reading a shared staging or handoff buffer after another re-staged it, which
// a lock around predict() alone would allow.
TEST_F(ProposalInferenceParityTest, SessionsOnOneSharedPairDoNotCrosstalk) {
  const EvidenceCase* partial = nullptr;
  for (const EvidenceCase& c : cases_)
    if (c.name == "partial") partial = &c;
  ASSERT_NE(partial, nullptr);

  // Nested candidate subsets, each with the partial case's evidence that falls
  // inside it.
  std::shared_ptr<MoveProposalNets> nets = MoveProposalNets::create(nets_params(num_moves_));
  const int subsets[] = {num_moves_, 40, 25, 10};
  std::vector<ConsumerCase> consumers;
  for (int subset : subsets) {
    ASSERT_LE(subset, num_moves_);
    ConsumerCase c;
    c.moves = truncate_moves(moves_, subset);
    for (int idx : partial->indices)
      if (idx < subset) c.evidence.add(sobs_moves_[idx], obs_[idx], idx);
    MoveProposalSession solo(nets);
    c.want_plain = solo.encode(board_.data(), c.moves);
    c.want_conditioned = solo.condition(c.evidence);
    consumers.push_back(std::move(c));
  }

  MoveProposalSession a(nets);
  MoveProposalSession b(nets);
  a.encode(board_.data(), consumers[0].moves);
  const MoveProposalPredictions got_b_plain = b.encode(board_.data(), consumers[1].moves);
  a.condition(EvidenceSet{});
  const MoveProposalPredictions got_b = b.condition(consumers[1].evidence);
  const MoveProposalPredictions got_a = a.condition(consumers[0].evidence);
  expect_same_predictions(got_a, consumers[0].want_conditioned, kFp32Tol, "session a");
  expect_same_predictions(got_b_plain, consumers[1].want_plain, kFp32Tol, "session b plain");
  expect_same_predictions(got_b, consumers[1].want_conditioned, kFp32Tol, "session b");

  // One thread and session per consumer.
  const int iterations = 8;
  std::vector<ConsumerRun> runs;
  for (const ConsumerCase& c : consumers) runs.push_back(ConsumerRun{&c, {}});
  std::vector<std::thread> threads;
  for (ConsumerRun& r : runs)
    threads.emplace_back(&ConsumerRun::run, &r, nets, board_.data(), iterations);
  for (std::thread& t : threads) t.join();
  for (size_t i = 0; i < runs.size(); ++i) {
    std::cout << "  [thread " << i << ", M=" << subsets[i] << "] worst prob " << runs[i].worst.prob
              << ", sd " << runs[i].worst.score_diff << ", gain " << runs[i].worst.gain << "\n";
    EXPECT_LE(runs[i].worst.prob, kFp32Tol.prob) << "thread " << i;
    EXPECT_LE(runs[i].worst.score_diff, kFp32Tol.score_diff) << "thread " << i;
    EXPECT_LE(runs[i].worst.gain, kFp32Tol.gain) << "thread " << i;
  }
}

// Equal engine-determining params share one live pair, even when callers race
// (later ones wait out the first build); a different row bound gets its own.
TEST_F(ProposalInferenceParityTest, CreateSharesAPairAcrossEqualParams) {
  std::vector<std::shared_ptr<MoveProposalNets>> racers(6);
  std::vector<std::thread> threads;
  for (std::shared_ptr<MoveProposalNets>& slot : racers) {
    threads.emplace_back([&] { slot = MoveProposalNets::create(nets_params(num_moves_)); });
  }
  for (std::thread& t : threads) t.join();
  for (const std::shared_ptr<MoveProposalNets>& r : racers) EXPECT_EQ(r.get(), racers[0].get());
  std::shared_ptr<MoveProposalNets> again = MoveProposalNets::create(nets_params(num_moves_));
  EXPECT_EQ(again.get(), racers[0].get());
  std::shared_ptr<MoveProposalNets> other = MoveProposalNets::create(nets_params(32));
  EXPECT_NE(other.get(), racers[0].get());
}

// A cache and step graph from different checkpoints of the same architecture
// pass every shape and layout check, and would serve plausible wrong numbers.
// Only the proposal_export_id fingerprint catches them, and this is the one
// test of that guard. step_mismatch.onnx is the other checkpoint's step graph.
TEST_F(ProposalInferenceParityTest, RejectsACacheStepPairFromDifferentModels) {
  MoveProposalNets::Params params = nets_params(num_moves_);
  params.step_onnx_path = model("step_mismatch.onnx");
  try {
    MoveProposalNets::create(params);
    ADD_FAILURE() << "loading a cache/step pair from different models should have been rejected";
  } catch (const std::runtime_error& e) {
    // Match the cause, so a load that failed for another reason cannot pass.
    EXPECT_NE(std::string(e.what()).find("different models"), std::string::npos) << e.what();
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
