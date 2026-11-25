#pragma once

// A stratified sample of a position's legal moves, the sim-labeled-candidates
// survey's "stratified" recipe (apps/sim_candidate_survey_tool.cpp): a handful
// of candidates dense at the head of the equity ranking, a slice of the
// contention zone just below it, a uniform tail (where surprising constructive
// plays live), and exchanges. It was the move-set model's distillation sample,
// whence the namespace.
//
// It takes `ranked`, the position's legal moves in descending
// static-equity order (equity_top_k with no cap).

#include "game/move.h"

#include <random>
#include <span>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// A position's selected candidates in storage order, and num_legal_moves, 0
// for a stratified sample.
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

}  // namespace move_set_eval
}  // namespace scribblez
