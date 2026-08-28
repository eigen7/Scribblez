// Evidence trajectories (docs/roadmap.md item 4): running one decision point's
// candidate selection (the recipe lives in evidence_trajectory_select.h)
// through the student scorer and the sim runner. The selected candidates are
// simmed together in one SimRunner call under common random numbers; each
// record carries its evidence role (SimObsRole), so a reader recovers which
// candidates are evidence-eligible from the role rather than the order. The
// proposer cannot yet condition on evidence mid-trajectory (roadmap item 3
// lands the fusion runtime), so the proposal distribution is computed once per
// position -- the roadmap's bootstrap proposer.
//
// This is the position-level core; the front-ends that supply decision points
// (.slog replay, .gcg position sets) live in the evidence_trajectory_generator
// app.
#pragma once

#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "nn/trt_eval_service.h"
#include "sim/sim_runner.h"
#include "training/evidence_trajectory_select.h"
#include "training/move_set_encoder.h"
#include "util/math.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <vector>

namespace scribblez::evidence {

using StudentService = nn::TrtEvalService<nn::MoveSetEvaluationSpec>;

// Throws util::CleanException on an unusable configuration (also validates
// the runner params). Call before any worker thread exists.
void validate(const TrajectoryOptions& opt);

// Parallelism is across positions, so each runner is single-threaded (see
// sim_obs_tool for the determinism rationale). `leaf` is the shared
// (serialized) truncation leaf service; null iff opt.horizon is 0.
SimRunner::Params sim_params(const TrajectoryOptions& opt, nn::PositionEvalService* leaf);

// The decision point a trajectory is run at.
struct DecisionPoint {
  SimPosition pos;              // board, scores, mover, rack; opp_leave under open leaves
  const GameStateEncoder* enc;  // replayed to the position: student input, score diff
  int bag_size;                 // the mover's-POV bag, for the equity ranking
};

struct TrajectoryResult {
  uint32_t num_legal_moves = 0;
  std::vector<Move> candidates;   // trajectory order: anchor, on-policy, off-policy
  std::vector<SimObsRole> roles;  // parallel to candidates
  std::vector<SimObservation> observations;
};

// Serializes every student evaluation onto one dedicated thread -- the
// NeuralNet contract is one net driven from one thread -- while position
// workers block on their request's completion. The sims are the long pole by
// orders of magnitude, so the round-trip costs nothing.
class StudentScorer {
 public:
  explicit StudentScorer(StudentService* service) : service_(service) {}

  // Blocks until `wld_out` / `sd_out` are filled for the batch. Called from
  // position workers.
  void score(const float* board_row, const move_set::MoveFeatureArrays* moves, float* wld_out,
             float* sd_out);

  // Thread body; returns once stop() was called and the queue is drained.
  void run();
  void stop();

 private:
  struct Request {
    const float* board_row;
    const move_set::MoveFeatureArrays* moves;
    float* wld_out;
    float* sd_out;
    bool done = false;
  };

  StudentService* service_;
  std::mutex mutex_;
  std::condition_variable queue_cv_;
  std::condition_variable done_cv_;
  std::deque<Request*> queue_;
  bool stopping_ = false;
};

// Per-worker: owns the scoring buffers, a single-threaded SimRunner and the
// proposal sampler; student evaluations round-trip through the shared scorer.
// The information condition is the spec's: with opp_leave_input the position's
// opp_leave is visible to the ranking and the student, else nothing is (the
// hidden mode's replayed opponent rack is ground truth the mover cannot see).
class TrajectoryRunner {
 public:
  // `leaf` is the shared (serialized) truncation leaf service; null iff
  // opt.horizon is 0.
  TrajectoryRunner(const Dictionary& dict, const InputEncodingSpec& spec,
                   const TrajectoryOptions& opt, StudentScorer* scorer,
                   nn::PositionEvalService* leaf = nullptr);

  // Rank, score, select, sim. `base_seed` feeds SimRunner directly; the
  // trajectory draws come from their own stream derived from it, so adding a
  // proposal never perturbs the rollout seeds.
  TrajectoryResult run(const DecisionPoint& dp, uint64_t base_seed);

 private:
  // The student's per-candidate win equities -- the proposal scores.
  const std::vector<float>& win_equities(const DecisionPoint& dp, const Rack& visible_opp,
                                         const std::vector<Move>& ranked);

  const Dictionary& dict_;
  InputEncodingSpec spec_;
  TrajectoryOptions opt_;
  StudentScorer* scorer_;
  SimRunner runner_;
  util::SoftmaxSampler sampler_;
  std::vector<float> board_row_;
  move_set::MoveFeatureArrays move_features_;
  std::vector<float> wld_buf_;
  std::vector<float> sd_buf_;
  std::vector<float> win_equity_;
};

}  // namespace scribblez::evidence
