// Evidence trajectories (docs/roadmap.md item 4): runs one decision point's
// candidate selection (evidence_trajectory_select.h) through the student scorer
// and the sim runner. The selected candidates are simmed together in one
// SimRunner call, under common random numbers. The result tags each candidate
// with its evidence role (SimObsRole), so a reader tells evidence-eligible
// candidates apart by role rather than by position in the list.
//
// Proposals are scored once per position by the plain student and are not
// re-conditioned on the sims as they land: this is the roadmap's generation-0
// proposer.
//
// This is the position-level core. The front-ends that supply decision points
// (.slog replay, .gcg position sets) live in apps/evidence_trajectory_generator.cpp.
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

// Throws util::CleanException on an unusable configuration, SimRunner params
// included. Call before any worker thread exists: TrajectoryRunner builds its
// SimRunner inside the worker, where a constructor throw would terminate the
// process instead of printing an error.
void validate(const TrajectoryOptions& opt);

// Parallelism is across positions, so each runner is single-threaded: that uses
// cores better than within-position threading and keeps every position's sims
// independent of the worker count. `leaf` is the shared truncation leaf service
// (it serializes its callers); null iff opt.horizon is 0.
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

// Runs every student evaluation on one dedicated thread, which serializes the
// NeuralNet's one-call-at-a-time contract. Position workers block until their
// request completes; the sims dominate the runtime by orders of magnitude, so
// the round trip costs nothing.
class StudentScorer {
 public:
  explicit StudentScorer(StudentService* service) : service_(service) {}

  // Called from position workers; blocks until `wld_out` and `sd_out` are filled.
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

// One per worker thread: owns the scoring buffers, a single-threaded SimRunner
// and the proposal sampler, and sends student evaluations through the shared
// scorer. spec.opp_leave_input sets the information condition. When it is on,
// the equity ranking and the student both see the position's opp_leave; when
// off, they see no opponent tiles at all, since in hidden mode the replayed
// opponent rack is ground truth the mover cannot see.
class TrajectoryRunner {
 public:
  // `leaf` as for sim_params().
  TrajectoryRunner(const Dictionary& dict, const InputEncodingSpec& spec,
                   const TrajectoryOptions& opt, StudentScorer* scorer,
                   nn::PositionEvalService* leaf = nullptr);

  // Rank, score, select, sim. `base_seed` seeds SimRunner directly; the
  // selection draws use a separate stream derived from it, so changing how
  // many proposals are drawn never perturbs the rollout seeds.
  TrajectoryResult run(const DecisionPoint& dp, uint64_t base_seed);

 private:
  // The student's per-candidate win equities, P(win) + P(draw)/2: the scores
  // the on-policy proposals are drawn from.
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
