#include "sim/macondo_autostopper.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace scribblez {

namespace {

// Macondo's defaults. The iteration cap grows with the rollout depth.
constexpr int kIterationsCutoff = 2000;
constexpr int kPerPlyStopScaling = 625;
constexpr uint64_t kSimilarPlaysIterationsCutoff = 750;
constexpr double kMinReasonableWinProb = 0.005;
// Pruning waits for this many samples, so standard errors are trustworthy.
constexpr int kMinIterationsForPruning = 128;
// Stop99: the total probability of wrongly pruning the best play, and the
// z-score of the equity-tiebreak tests.
constexpr double kDelta = 0.01;
constexpr double kZ99 = 2.576;
// Win probabilities closer than this tie, and equity decides.
constexpr double kWinProbEpsilon = 1e-9;

bool ranks_ahead(const SimmedPlay& a, const SimmedPlay& b) {
  const double wa = a.win_prob.mean(), wb = b.win_prob.mean();
  if (std::abs(wa - wb) > kWinProbEpsilon) return wa > wb;
  return a.equity.mean() > b.equity.mean();
}

// Macondo's zTest: whether the sample mean `m`, with standard error `e`, lies
// more than `z` standard errors above `mu` (below, with `below` set). A zero
// standard error yields an infinite or NaN statistic, as in Go.
bool z_test(double mu, double m, double e, double z, bool below) {
  const double zcalc = (m - mu) / e;
  return below ? z > zcalc : zcalc > z;
}

// Indices of `plays`, best first, pruned plays last in their current order.
std::vector<int> pruning_order(const std::vector<SimmedPlay>& plays) {
  std::vector<int> order(plays.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (plays[a].ignored || plays[b].ignored) return !plays[a].ignored && plays[b].ignored;
    return ranks_ahead(plays[a], plays[b]);
  });
  return order;
}

// Whether the leader's win probability is confidently near 0, or both it and
// the last unpruned play's are confidently near 1. Win probability then cannot
// separate the plays, and equity ranks them instead.
bool tiebreak_by_equity(const SimmedPlay& leader, const SimmedPlay& bottom) {
  const double mu = leader.win_prob.mean(), e = leader.win_prob.standard_error();
  if (z_test(kMinReasonableWinProb, mu, e, -kZ99, /*below=*/true)) return true;
  return z_test(1 - kMinReasonableWinProb, mu, e, kZ99, /*below=*/false) &&
         z_test(1 - kMinReasonableWinProb, bottom.win_prob.mean(), bottom.win_prob.standard_error(),
                kZ99, /*below=*/false);
}

// Moves the play with the highest mean equity, pruned or not, to the front.
void promote_equity_leader(const std::vector<SimmedPlay>& plays, std::vector<int>& order) {
  double highest = -1000000.0;
  int best = -1;
  for (int i = 0; i < int(order.size()); ++i) {
    const double eq = plays[order[i]].equity.mean();
    if (eq > highest) {
      highest = eq;
      best = i;
    }
  }
  if (best > 0) std::swap(order[0], order[best]);
}

}  // namespace

void rank_simmed_plays(std::vector<SimmedPlay>& plays) {
  std::stable_sort(plays.begin(), plays.end(), [](const SimmedPlay& a, const SimmedPlay& b) {
    if (a.ignored != b.ignored) return !a.ignored;
    return ranks_ahead(a, b);
  });
}

bool MacondoAutoStopper::should_stop(uint64_t iterations, std::vector<SimmedPlay>& plays,
                                     int plies) const {
  const int n = plays.size();
  if (n < 2) return true;
  if (iterations > uint64_t(kIterationsCutoff + plies * kPerPlyStopScaling)) return true;
  const int ignored =
    std::count_if(plays.begin(), plays.end(), [](const SimmedPlay& p) { return p.ignored; });
  if (ignored >= n - 1) return true;

  std::vector<int> order = pruning_order(plays);
  const SimmedPlay& bottom = plays[order[n - ignored - 1]];
  const bool by_equity = tiebreak_by_equity(plays[order[0]], bottom);
  if (by_equity) promote_equity_leader(plays, order);
  const SimmedPlay& leader = plays[order[0]];
  const util::RunningStat& leader_stat = by_equity ? leader.equity : leader.win_prob;

  // Prune play i when its upper confidence bound falls below the leader's
  // lower one. c is a Bonferroni correction over the K unpruned plays: each
  // gets a delta / K error budget, so the chance of any false prune is at most
  // delta. Pruning shrinks K and so tightens later checks.
  const double k = std::max(2, n - ignored);
  const double c = std::sqrt(2 * std::log(k / kDelta));
  const double leader_lcb = leader_stat.mean() - c * leader_stat.standard_error();

  int newly_ignored = 0;
  for (int i = 1; i < n; ++i) {
    SimmedPlay& p = plays[order[i]];
    if (p.ignored) continue;
    const util::RunningStat& stat = by_equity ? p.equity : p.win_prob;
    if (stat.count() < kMinIterationsForPruning) continue;
    const bool trails = leader_lcb > stat.mean() + c * stat.standard_error();
    if (trails ||
        (iterations > kSimilarPlaysIterationsCutoff && materially_similar(leader.move, p.move))) {
      p.ignored = true;
      ++newly_ignored;
    }
  }
  return ignored + newly_ignored >= n - 1;
}

bool MacondoAutoStopper::materially_similar(const Move& a, const Move& b) const {
  if (a.type() != MoveType::PLAY || b.type() != MoveType::PLAY) return false;
  if (a.horizontal() != b.horizontal() || a.num_glyphs() != b.num_glyphs()) return false;
  if (a.word_origin(board_) != b.word_origin(board_)) return false;
  if (a.main_word(board_).size() != b.main_word(board_).size()) return false;
  std::vector<uint8_t> ga, gb;
  for (int i = 0; i < a.num_glyphs(); ++i) {
    ga.push_back(a.glyph(i).code());
    gb.push_back(b.glyph(i).code());
  }
  std::sort(ga.begin(), ga.end());
  std::sort(gb.begin(), gb.end());
  return ga == gb;
}

}  // namespace scribblez
