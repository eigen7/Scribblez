#pragma once

// Which of a position's legal moves the move-set-eval target generator labels,
// in the order it stores them (docs/roadmap.md, track A).
//
// Two selections, one per .mset kind:
//
//   * stratified_candidates -- the training sample. A handful of candidates
//     balancing the filter's failure modes: dense at the head of the equity
//     ranking (ranking precision), a slice of the contention zone, a uniform
//     tail (junk rejection, and where surprising constructive plays live), and
//     exchanges (the exchange head starves otherwise). Distillation needs
//     coverage, not unbiasedness -- the sampler only has to visit a move for
//     the teacher to value it honestly.
//
//   * full_sweep_candidates -- the evaluation slice. Every legal candidate,
//     capped, for the A3 gate metrics (top-K recall and teacher-value regret),
//     which a ~15-candidate sample structurally cannot measure: it never sees
//     the tail moves the filter exists to catch.
//
// Both take `ranked`, the position's legal moves in descending static-equity
// order (equity_top_k with no cap).

#include "game/move.h"

#include <random>
#include <span>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// A position's selected candidates in storage order, plus the count a .mset's
// TargetPositionHeader::num_legal_moves records: the moves the selection drew
// from, or 0 for the stratified sample, whose size says nothing about the
// position's legal-move count. A sweep reached everything iff
// candidates.size() equals it.
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

// The stratified sample, storing the actually-played move first (so a reader
// can recover the incumbent's own choice), then any `forced` moves not already
// present, then the ranking's head, then the sampled strata. `forced` carries
// the position's simmed trajectory candidates (docs/roadmap.md item 4): the
// value-labeled subset must always include them, or dense value labels stay at
// the static strata's rate while the proposer explores elsewhere.
Selection stratified_candidates(const std::vector<Move>& ranked, const Move& played,
                                const StratumQuotas& quotas, std::mt19937_64& rng,
                                std::span<const Move> forced = {});

// The off-policy floor for an evidence trajectory (docs/roadmap.md item 4):
// `count` distinct legal-move indices drawn uniformly from those the anchor and
// on-policy picks have not already taken (marked in *taken), returned in draw
// order and marked in turn. A plain uniform draw over the whole move list -- the
// bounded exploration floor against the proposer's echo chamber, deliberately
// assuming nothing about which moves are worth exploring (exchanges and the tail
// are sampled at their natural frequency in `ranked`, no stratum reserved for
// them). These draws are labels-only: a training row never places them in an
// evidence set. `taken` must be sized to ranked.size().
std::vector<size_t> off_policy_draws(const std::vector<Move>& ranked, int count,
                                     std::mt19937_64& rng, std::vector<char>* taken);

// The capped full sweep: the top `cap` candidates by static equity, plus every
// exchange candidate and the played move wherever they rank, in equity-rank
// order throughout.
//
// The cap exists for the two-blank racks that generate 20k+ legal moves --
// overwhelmingly redundant blank-designation variants -- whose sweep would cost
// more than the rest of the file combined; a cap in the low thousands leaves
// normal positions (a few hundred moves) complete, and the teacher's best move
// essentially never sits below equity rank ~2000. Truncation is not hidden: the
// caller records the position's legal-move count in the .mset, so a reader sees
// what the sweep did not reach.
//
// Exchanges and the played move are kept unconditionally because the metrics
// read them: the exchange head is scored on candidates static equity buries,
// and the played move is what the incumbent baseline ranks first. Keeping the
// selection a subsequence of `ranked` is what makes stored order *be* the
// static-equity ranking, so the incumbent baseline is exact at every K rather
// than a floor.
Selection full_sweep_candidates(const std::vector<Move>& ranked, const Move& played, int cap);

}  // namespace move_set_eval
}  // namespace scribblez
