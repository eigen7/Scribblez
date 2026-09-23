#pragma once

// Which of a position's legal moves the .mset target generator labels, and in
// what stored order. There are two selections, one per .mset kind:
//
//   * stratified_candidates: the training sample. A handful of candidates
//     spread over the student's failure modes: dense at the head of the equity
//     ranking (ranking precision), a slice of the contention zone just below
//     it, a uniform tail (junk rejection, and where surprising constructive
//     plays live), and exchanges (which the exchange head needs to see).
//     Distillation needs coverage, not unbiasedness: the teacher values any
//     move it is shown honestly.
//
//   * full_sweep_candidates: the evaluation slice. Every legal candidate, up to
//     a cap, for the top-K recall and teacher-value regret metrics
//     (docs/move_set_eval_results.md). A ~15-candidate sample cannot measure
//     those, because it never sees the tail moves the student must learn to
//     reject.
//
// Every function here takes `ranked`, the position's legal moves in descending
// static-equity order (equity_top_k with no cap).

#include "game/move.h"

#include <random>
#include <span>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// A position's selected candidates in storage order, plus the value for
// TargetPositionHeader::num_legal_moves: the legal-move count for a sweep, 0 for
// a stratified sample.
struct Selection {
  std::vector<Move> candidates;
  uint32_t num_legal_moves;
};

// Per-stratum candidate counts for stratified_candidates.
struct StratumQuotas {
  int top = 4;              // candidates from the head of the ranking
  int mid = 4;              // sampled from ranks [top, mid_rank_limit)
  int tail = 4;             // sampled from ranks [mid_rank_limit, n)
  int exchange = 2;         // sampled among the non-PLAY candidates
  int mid_rank_limit = 32;  // exclusive rank bound of the contention zone
};

// The stratified sample, in storage order: the move actually played (so a
// reader can recover the incumbent's choice), then any `forced` moves not
// already present, then the ranking's head, then the sampled strata. `forced`
// carries the position's simmed trajectory candidates (docs/roadmap.md item 4),
// which must always get value labels; otherwise dense labels would stay on the
// static strata while the proposer explores elsewhere. The head quota counts
// after the forced moves, so forcing adds candidates rather than displacing the
// head.
Selection stratified_candidates(const std::vector<Move>& ranked, const Move& played,
                                const StratumQuotas& quotas, std::mt19937_64& rng,
                                std::span<const Move> forced = {});

// The off-policy floor of an evidence trajectory (evidence_trajectory_select.h):
// up to `count` distinct indices into `ranked`, drawn uniformly from those not
// yet marked in *taken, returned in draw order and marked as drawn. `taken` is
// sized to ranked.size(). The floor bounds the proposer's echo chamber without
// assuming which moves are worth exploring: a uniform draw reaches exchanges
// and the tail at their natural frequency, with no stratum reserved for them.
std::vector<size_t> off_policy_draws(const std::vector<Move>& ranked, int count,
                                     std::mt19937_64& rng, std::vector<char>* taken);

// The capped full sweep: the top `cap` candidates by static equity, plus every
// exchange and the played move wherever they rank, all in equity-rank order.
//
// The cap exists for two-blank racks, which generate 20k+ legal moves (mostly
// redundant blank designations) and would cost more than the rest of the file
// combined. A cap in the low thousands leaves normal positions (a few hundred
// moves) complete, and the teacher's best move essentially never ranks below
// ~2000. Truncation stays visible: num_legal_moves records the full count.
//
// Exchanges and the played move are kept regardless of rank because the metrics
// read them: the exchange head is scored on candidates static equity buries,
// and the played move is the incumbent baseline's top choice. Keeping the
// selection a subsequence of `ranked` makes the stored order the static-equity
// ranking, so the incumbent baseline is exact at every K rather than a bound.
Selection full_sweep_candidates(const std::vector<Move>& ranked, const Move& played, int cap);

}  // namespace move_set_eval
}  // namespace scribblez
