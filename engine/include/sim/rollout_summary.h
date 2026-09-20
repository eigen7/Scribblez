#pragma once

// A compact reduction of one candidate's rollouts, for analysis rather than
// training: where a SimObservation keeps per-footprint placement histograms
// (35 KB), this keeps what explains WHY a candidate sims as it does -- how the
// final margin is distributed, and what each side's next move scored and
// whether it played off the candidate's own tiles. A candidate that sims well
// because the opponent's replies score less is defending; one whose own next
// move scores more, off its own tiles, is setting up; one that only reshapes
// the margin distribution is trading variance.

#include "game/move.h"
#include "sim/sim_runner.h"

#include <array>
#include <cstdint>
#include <span>

namespace scribblez {

// A next move's score, bucketed by tens: bin b counts scores in [10b, 10b + 10),
// the last bin everything from 100 up.
inline constexpr int kScoreBins = 11;
inline constexpr int kScoreBinWidth = 10;

// The final margin (mover minus opponent), bucketed by 25s over [-200, 200):
// bin 0 is everything below -200, the last bin everything from 200 up.
inline constexpr int kDeltaBins = 18;
inline constexpr int kDeltaBinWidth = 25;
inline constexpr int kDeltaBinFloor = -200;

// The end-of-game rack settlement's swing (end_rack_swing), bucketed by 10s over
// [-50, 50): bin 0 is everything below -50, the last bin everything from 50 up.
inline constexpr int kEndSwingBins = 12;
inline constexpr int kEndSwingBinWidth = 10;
inline constexpr int kEndSwingBinFloor = -50;

// One side's next move after the candidate, over the candidate's rollouts.
struct NextMoveStats {
  double score_sum = 0;
  std::array<uint32_t, kScoreBins> score_hist{};
  uint32_t bingos = 0;     // plays of all seven tiles
  uint32_t non_plays = 0;  // exchanges, passes, and rollouts that ended first
  // Plays that laid a tile on a square orthogonally adjacent to one the
  // candidate placed: a hook on it, a play through it, or a parallel beside it.
  uint32_t adjacent = 0;
};

struct RolloutSummary {
  uint32_t n = 0;
  double wins = 0;
  double draws = 0;
  double losses = 0;
  double delta_sum = 0;
  double delta_sq_sum = 0;
  std::array<uint32_t, kDeltaBins> delta_hist{};
  NextMoveStats opp_reply;
  NextMoveStats self_next;
  // How the games ended: the tile values left on each rack (a candidate that
  // strands the opponent's Q shows up in opp_stranded_sum), who played out, and
  // the settlement's swing on the final margin.
  double self_stranded_sum = 0;
  double opp_stranded_sum = 0;
  uint32_t self_went_out = 0;
  uint32_t opp_went_out = 0;
  double end_swing_sum = 0;
  std::array<uint32_t, kEndSwingBins> end_swing_hist{};
};

// The paired difference in win value (win = 1, draw = 1/2) between two
// candidates over the same rollout indices. Under common random numbers rollout
// i of both faced the same opponent rack, so these moments give the standard
// error of the candidates' win-rate difference exactly, where the two marginal
// errors would overstate it.
struct PairedWinDiff {
  double sum = 0;     // of (a - b)
  double sq_sum = 0;  // of (a - b)^2
};

// True iff `d`, over `n` paired rollouts, puts a's win rate more than `sigmas`
// standard errors BELOW b's: the early-stopping test of a racing sim.
bool clearly_below(const PairedWinDiff& d, size_t n, double sigmas);

PairedWinDiff paired_win_diff(std::span<const RolloutResult> a, std::span<const RolloutResult> b);

// Reduce `candidate`'s rollouts, in order.
RolloutSummary summarize_rollouts(const Move& candidate, std::span<const RolloutResult> rollouts);

}  // namespace scribblez
