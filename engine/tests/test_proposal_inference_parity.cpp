// The move-proposal two-graph runtime (agent/move_proposal_nets.h +
// agent/move_proposal_session.h) against the PyTorch model it serves (roadmap
// item 3). mset_infer_parity's counterpart for the split evidence-path graphs.
//
// A session runs the cache graph once, then -- for each evidence set the
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
// The cache graph's planes -- the evidence-free predicted planes every evidence
// token's predicted half is gathered from -- are checked on the session's
// retained cache against the plain reference; the step graph emits none.
//
// The fixture (one model's cache/step pair, a candidate set, its raw Move /
// SimObservation records, and several evidence cases -- empty, partial with
// scattered+duplicate indices, and full at the padding boundary) is generated
// by py/scripts/move_set_eval/gen_proposal_parity_fixture.py, invoked at the
// build-time SCRIBBLEZ_PY_DIR; the test skips when that generator cannot run
// (no torch/onnx). Pass a fixture directory as the first non-gtest argument to
// reuse one:
//   test_proposal_inference_parity <fixture_dir>

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

// The fixture's per-candidate reference scalars: [p_win, p_draw, p_loss,
// sd_mean, sd_std, gain]. The plane reference is a separate M x 52 x 225 file
// (the cache graph's footprint slot-channel planes output).
constexpr int kScalarFields = 6;
constexpr int kPlaneFloats = scribblez::nn::PlanesOutput::kRowElems;

// How far a TensorRT run may deviate from the PyTorch FP32 reference. Set from
// the measured noise floor, not loosely: on this fixture the observed worst
// deviations are ~1e-5 (wld probs), ~3.5e-4 (planes, a 900-cell softmax-then-
// per-cell-marginal off a 225-token attention -- the jitteriest head), and
// ~5e-5 (score_diff / gain).
// The bounds sit above those with room for cross-GPU / cross-build kernel jitter
// -- ~6x on the jittery planes head, a wider (~50-100x) round-number margin on
// the far tighter wld / score_diff / gain -- yet stay orders of magnitude below
// any real defect (a dropped ev_obs field, a mis-strided evidence plane, a
// mis-bound handoff move outputs by far more). The test prints the actual
// deviations, so retune here if a future model legitimately needs it.
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

// One evidence case as the fixture describes it.
struct EvidenceCase {
  std::string name;
  std::vector<int> indices;        // scored indices, empty for the empty case
  std::vector<float> ref_scalars;  // M x kScalarFields
};

// The worst deviation of a prediction from the reference, per field group. NaN
// is caught rather than hidden (std::max returns its first argument on a NaN
// comparison, so an all-NaN run -- what an unbound buffer looks like -- would
// otherwise report a perfect match).
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

  // The params of a loaded pair over the fixture's graphs, its plan cache in
  // this test's scratch root, engines bounded at `max_rows` (below the
  // candidate count to exercise chunking of both graphs).
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
  params.precision = scribblez::nn::Precision::kFP32;  // item-3 serving precision
  params.max_rows = max_rows;
  params.step_max_rows = max_rows;  // so a bound below M chunks BOTH graphs
  params.mount_root = cache_root_.string();
  // The parity check validates the inference stack, not kernel-tactic quality,
  // so build at optimization level 0 to keep cold builds to a few seconds.
  params.fast_build = true;
  return params;
}

EvidenceSet ProposalInferenceParityTest::evidence_of(const EvidenceCase& c) const {
  EvidenceSet evidence;
  for (int idx : c.indices) evidence.add(sobs_moves_[idx], obs_[idx], idx);
  return evidence;
}

// Every candidate's conditioned prediction against the reference, for one case.
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

// The session's retained cache planes against the plain reference.
void expect_planes_match(const std::vector<float>& got, const std::vector<float>& ref,
                         int num_moves, Tolerance tol) {
  ASSERT_EQ(got.size(), size_t(num_moves) * kPlaneFloats);
  Worst worst;
  for (size_t i = 0; i < got.size(); ++i)
    track(worst.planes, got[i], ref[i], "planes", int(i / kPlaneFloats));
  std::cout << "  [cache planes] max err = " << worst.planes << " (tol " << tol.planes << ")\n";
  EXPECT_LE(worst.planes, tol.planes);
}

// Two predictions over the same candidates, held to `tol` on every field --
// what one session's outputs must equal whether or not another session used
// the nets in between.
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

// The full run at a single chunk: every candidate scored in one predict() per
// graph.
TEST_F(ProposalInferenceParityTest, MatchesPyTorchReferenceForEveryEvidenceCase) {
  MoveProposalSession session(MoveProposalNets::create(nets_params(num_moves_)));
  const MoveProposalPredictions plain = session.encode(board_.data(), moves_);
  ASSERT_TRUE(plain.gain.empty()) << "the cache graph emits no gain head";
  expect_planes_match(session.cache().planes, plain_planes_, num_moves_, kFp32Tol);

  for (const EvidenceCase& c : cases_) {
    const MoveProposalPredictions& conditioned = session.condition(evidence_of(c));
    expect_case_matches(conditioned, c, num_moves_, kFp32Tol);

    if (c.name == "empty") {
      // The step graph at an empty set must reproduce the cache's plain heads
      // (and the plain mset outputs, which the reference IS), within tolerance.
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

// The same candidate set scored with the engines bounded below M, so both the
// cache and step graphs run in chunks and the session's full-M retention of
// move_enc / board / g across chunk boundaries is exercised.
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

// One consumer's worth of work over a shared pair: a candidate subset and the
// evidence that fits it, with what that consumer produces when nothing else
// touches the nets.
struct ConsumerCase {
  scribblez::move_set::MoveFeatureArrays moves;
  EvidenceSet evidence;
  MoveProposalPredictions want_plain;
  MoveProposalPredictions want_conditioned;
};

// The worst deviation over one consumer's outputs from its solo reference,
// across `iterations` rounds of encode / condition(empty) / condition -- run on
// its own thread, so it reports rather than asserts.
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

// Sessions over ONE shared pair, their encode/condition calls interleaved over
// different candidate sets and evidence, each reproduce what they produce
// alone: the nets hold no per-position state, and the mutex spans a whole
// call. First interleaved on one thread (the sequencing), then from real
// concurrent threads (the locking): a leak of one session's cache into
// another's step -- a shared handoff or staging buffer read after another
// session re-staged it, which a critical section narrowed to predict() alone
// would allow -- moves outputs by far more than the parity tolerance.
TEST_F(ProposalInferenceParityTest, SessionsOnOneSharedPairDoNotCrosstalk) {
  const EvidenceCase* partial = nullptr;
  for (const EvidenceCase& c : cases_)
    if (c.name == "partial") partial = &c;
  ASSERT_NE(partial, nullptr);

  // Four consumers over nested candidate subsets (the full set first), each
  // with the partial case's evidence that fits its subset, and their solo
  // references.
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

  // Interleaved on one thread.
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

  // Concurrently: one thread per consumer, each its own session, hammering the
  // shared pair.
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

// create() dedupes on the full engine-determining params: equal params share
// one live pair -- also when the callers race, the second waiting out the
// first's build -- and a differing row bound gets its own.
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

// A cache and step graph exported from different checkpoints (same architecture,
// so C and the layout checks agree, but different weights) share no
// proposal_export_id -- create() must reject the pair loudly rather than serve
// plausible-looking wrong numbers. step_mismatch.onnx is that second model's
// step graph; the mismatch is caught only by the fingerprint, so this is the one
// test of that guard.
TEST_F(ProposalInferenceParityTest, RejectsACacheStepPairFromDifferentModels) {
  MoveProposalNets::Params params = nets_params(num_moves_);
  params.step_onnx_path = model("step_mismatch.onnx");
  try {
    MoveProposalNets::create(params);
    ADD_FAILURE() << "loading a cache/step pair from different models should have been rejected";
  } catch (const std::runtime_error& e) {
    // Matched against what it should object to, so a load that failed for some
    // other reason (a missing file, a layout mismatch) does not pass as this.
    EXPECT_NE(std::string(e.what()).find("different models"), std::string::npos) << e.what();
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) g_fixture_dir = argv[1];
  return RUN_ALL_TESTS();
}
