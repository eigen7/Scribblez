#pragma once

// The policy-set model's per-candidate labels from a raced sim
// (docs/plans/deep_bot_v0.md, "The pool record"): each candidate's mean win
// probability, its paired standard error against the leader, and q(a), the
// probability that it is the pool's best move.
//
// q(a) is the normal approximation to the paired differences against the
// leader, treating the non-leaders as independent of each other. Candidate a
// beats the leader with probability p_a = Phi(d_a / se_a), d_a its mean paired
// difference and se_a that mean's standard error. Then a is best with
// probability p_a * Prod_{b != a} (1 - p_b), and the leader with Prod_b
// (1 - p_b). Normalized, the common product cancels: q_a is proportional to
// p_a / (1 - p_a), and q_leader to 1. Racing stops weak candidates early, on
// fewer rollouts; their wider standard errors carry that.

#include "sim/sim_runner.h"

#include <vector>

namespace scribblez {

struct PoolLabel {
  double win_prob = 0;   // mean of p_win + p_draw / 2 over the candidate's rollouts
  double paired_se = 0;  // standard error of its paired difference against the leader; 0 for it
  double q = 0;          // probability that it is the pool's best move
  int rollouts = 0;      // completed; racing stops some early
};

// The labels of candidates whose rollouts are `rollouts[c]`, candidate c's
// rollout i the same deal for every candidate that reached it (common random
// numbers). The leader is the best mean among the candidates that completed
// the most rollouts, which racing never stops.
std::vector<PoolLabel> pool_labels(const std::vector<std::vector<Rollout>>& rollouts);

}  // namespace scribblez
