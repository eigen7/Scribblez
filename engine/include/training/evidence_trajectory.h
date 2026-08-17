// Evidence trajectories (docs/roadmap.md item 4): the deployment schedule's
// candidate selection at one decision point -- the greedy anchor, then a
// randomized-length sequence of proposals drawn from a temperature-softmax
// over the move set evaluation model's win equities, then one uniform-random
// draw over the remaining legal moves -- simmed under common random numbers.
// The uniform tail is appended LAST deliberately: training rows pair an
// evidence prefix with a held-out simmed candidate, so a last-slot sim yields
// proves-best labels at every prefix size while never entering an evidence
// set (the deployed loop's evidence contains only proposer picks). See
// docs/sim_residual_feedback.md, "Evidence-trajectory generation".
//
// All of a position's candidates are simmed in one SimRunner call, so the
// trajectory order is pure record bookkeeping: every prefix of a position's
// record array is a valid evidence set. The proposer cannot yet condition on
// evidence mid-trajectory (roadmap item 5 lands the fusion runtime), so the
// proposal distribution is computed once per position and sampled without
// replacement -- the roadmap's bootstrap proposer.
//
// This is the position-level core; the front-ends that supply decision points
// (.slog replay, .gcg position sets) live in the evidence_trajectory_generator
// app.
#pragma once

#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "nn/trt_eval_service.h"
#include "sim/sim_runner.h"
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

struct TrajectoryOptions {
  int rollouts = 200;
  int proposals_min = 2;
  int proposals_max = 8;
  double temperature = 0.05;  // win-equity units
  int proposal_pool = 64;     // proposals are drawn from the model's top-N unsimmed candidates
};

// Throws util::CleanException on an unusable configuration (also validates
// the runner params). Call before any worker thread exists.
void validate(const TrajectoryOptions& opt);

// Parallelism is across positions, so each runner is single-threaded (see
// sim_obs_tool for the determinism rationale).
SimRunner::Params sim_params(const TrajectoryOptions& opt);

// The decision point a trajectory is run at.
struct DecisionPoint {
  SimPosition pos;              // board, scores, mover, rack; opp_leave under open leaves
  const GameStateEncoder* enc;  // replayed to the position: student input, score diff
  int bag_size;                 // the mover's-POV bag, for the equity ranking
};

struct TrajectoryResult {
  uint32_t num_legal_moves = 0;
  bool uniform_tail = false;     // whether the last candidate is the uniform draw
  std::vector<Move> candidates;  // trajectory order
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

// The anchor: the highest-raw-score candidate, taken off the move list by a
// rule no model can be wrong about. `ranked` is descending static equity, so
// ties resolve to the equity-preferred instance deterministically.
size_t anchor_index(const std::vector<Move>& ranked);

// The trajectory's candidate indices into `ranked`, in sim order: anchor,
// then up to a sampled count of temperature-softmax proposals over the
// student's win equities, then (when any move remains) one uniform draw.
// Sets *uniform_tail accordingly.
std::vector<size_t> select_trajectory(const std::vector<Move>& ranked,
                                      const std::vector<float>& win_equity,
                                      const TrajectoryOptions& opt, std::mt19937_64& rng,
                                      util::SoftmaxSampler& sampler, bool* uniform_tail);

// Per-worker: owns the scoring buffers, a single-threaded SimRunner and the
// proposal sampler; student evaluations round-trip through the shared scorer.
// The information condition is the spec's: with opp_leave_input the position's
// opp_leave is visible to the ranking and the student, else nothing is (the
// hidden mode's replayed opponent rack is ground truth the mover cannot see).
class TrajectoryRunner {
 public:
  TrajectoryRunner(const Dictionary& dict, const InputEncodingSpec& spec,
                   const TrajectoryOptions& opt, StudentScorer* scorer);

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
