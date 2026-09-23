#pragma once

// Monte-Carlo sims over positions of a loaded .slog file: replay each position
// to its pre-move decision point, rank the legal moves by HastyBot static
// equity, select candidates, and sim them with SimRunner. Every tool that sims
// .slog positions without a model goes through here (sim_obs_tool's .sobs
// sidecars, the sim-labeled-candidates survey of
// docs/plans/sim_labeled_candidates.md), so they cannot drift on the replay,
// the information condition, or the seeding.

#include "data/slog_sampling.h"
#include "game/move.h"
#include "sim/rollout_summary.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_candidates.h"
#include "util/progress.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <vector>

namespace scribblez {

class Dictionary;

// Which of a position's equity-ranked legal moves get simmed: a stratified
// sample (move_set_eval::stratified_candidates) when `quotas` is set, else the
// top_k.
struct SimCandidateRecipe {
  int top_k = 10;
  std::optional<move_set_eval::StratumQuotas> quotas;
};

// A position's selected candidates. Empty `moves` means the selector declined
// the position, and it is not simmed.
struct SimCandidates {
  std::vector<Move> moves;
  // Parallel to `moves`: 0-based rank in the static-equity ranking, or -1 for a
  // move the ranking lacks (a PASS while other moves were legal).
  std::vector<int32_t> equity_ranks;
  // Parallel to `moves`: the candidates a selector is specifically asking
  // about, such as setup_selector's setup plays; all false otherwise.
  std::vector<char> highlighted;
  // Parallel to `moves`: HastyBot static equity. Filled by the simmer, not by
  // selectors.
  std::vector<double> equities;
  uint32_t num_legal_moves = 0;
};

// Apply `recipe` to `ranked` (every legal move, best equity first). `played`
// is the move the game made, which the stratified sample lists first.
SimCandidates select_sim_candidates(const std::vector<Move>& ranked, const Move& played,
                                    const SimCandidateRecipe& recipe, std::mt19937_64& rng);

// Chooses a position's candidates from `ranked` (every legal move, best equity
// first). `played` is the move the game made there; `rng` is seeded per
// position, so a selector is deterministic in (run seed, game, turn).
using SimCandidateSelector = std::function<SimCandidates(
  const binlog::GamePositionIndex& at, const SimPosition& pos, const std::vector<Move>& ranked,
  const Move& played, std::mt19937_64& rng)>;

SimCandidateSelector recipe_selector(const SimCandidateRecipe& recipe);

// Selects the top `cut` of the ranking plus up to `max_setups` of the
// best-ranked high-value setups (sim/setup_plays.h) below the cut, the setups
// highlighted. Declines a position with no setup below the cut, since there
// the cut hides nothing.
SimCandidateSelector setup_selector(const Dictionary& dict, int cut, int max_setups);

// Selects the top `cut` of the ranking plus the best-ranked `max_plays`
// (0 = all) plays below it that place no blank. Blank-placing plays are most
// of a blank rack's thousands of legal moves, nearly all of them designation
// variants of each other. High-value setups are highlighted. Never declines.
SimCandidateSelector all_plays_selector(const Dictionary& dict, int cut, int max_plays);

// Selects exactly the moves `chosen` names for a position, which must be legal
// there, and declines positions it does not name. For the confirm stage of a
// screen-then-confirm survey.
using ChosenMoves = std::map<binlog::GamePositionIndex, std::vector<Move>>;
SimCandidateSelector chosen_selector(ChosenMoves chosen);

struct SlogSimConfig {
  // Sim under face-up leaves: the opponent's kept tiles are known, to the
  // rollouts and to the candidate ranking.
  bool open_leaves = false;
  SimCandidateSelector selector = recipe_selector({});
  // Per-position runner params. `threads` here must be 1: parallelism is
  // across positions, which uses cores better than threading within one.
  SimRunner::Params runner;
  // Positions with at most this many unseen tiles (the bag plus the opponent's
  // rack) sim with runner.solve_endgames on; -1 = none. Solving costs several
  // times a greedy rollout and matters only when the candidate's consequences
  // reach the endgame, so it is spent where the game is about to end.
  int solve_endgames_max_unseen = -1;
  uint64_t seed = 0;  // run seed; each position sims under position_seed(seed, game, turn)
  // Added to each position's SimRunner seed and nothing else, so runs that
  // differ only here sim the same candidates with independent rollouts, as a
  // held-out estimate of a sim pick's value needs. Rollout i uses seed
  // base + i, so offsets must be at least a rollout count apart.
  uint64_t rollout_seed_offset = 0;
  int threads = 1;  // position workers
  // Reduce each candidate's rollouts to a RolloutSummary (sim/rollout_summary.h),
  // cheap enough to keep for every legal play, instead of a 35 KB
  // SimObservation.
  bool keep_summaries = false;
  // With summaries: the first this-many candidates are references, and every
  // candidate gets its paired win difference against each of them.
  int paired_references = 0;
  // With summaries: race the candidates instead of simming each to the full
  // count. `race_checkpoints` are ascending cumulative rollout counts. At each
  // one, a candidate whose win rate is more than `race_sigmas` paired standard
  // errors below the leader's stops, keeping the rollouts it got. The first
  // `race_protected` candidates always run to the end. Rollout i is the same
  // deal for every candidate that reaches it, so a survivor's rollouts are
  // exactly those of an unraced sim. Survival is a selection, though, and
  // biases the survivors' estimates upward; draw conclusions from a fresh sim.
  std::vector<int> race_checkpoints;
  double race_sigmas = 3.0;
  int race_protected = 0;
};

struct SimmedPosition {
  binlog::GamePositionIndex pos;
  SimPosition position;    // the replayed decision point
  uint64_t base_seed = 0;  // the SimRunner::run seed used
  SimCandidates candidates;
  int bag_size = 0;              // tiles in the bag at the decision point
  int unseen = 0;                // the bag plus the opponent's rack
  bool solved_endgames = false;  // the rollouts solved their endgames
  Move played;                   // the move the game made here
  // Parallel to candidates.moves; each filled per SlogSimConfig.
  std::vector<SimObservation> observations;
  std::vector<RolloutSummary> summaries;
  // paired[c][r]: candidate c's win value minus reference r's, over the rollouts.
  std::vector<std::vector<PairedWinDiff>> paired;
};

// Sim every position of `work` within the loaded .slog bytes `buf`, returning
// results in `work` order, independent of the thread count. Declined positions
// come back with empty candidates. A worker's failure rethrows here as a
// util::Exception naming the position ("game G turn T: ...").
std::vector<SimmedPosition> sim_slog_positions(const std::vector<char>& buf, const Dictionary& dict,
                                               const SlogSimConfig& config,
                                               const std::vector<binlog::GamePositionIndex>& work,
                                               util::ProgressMeter* meter);

// Candidate selection alone, without sims. Selection is deterministic, so a
// caller can scan a file for the positions a selector accepts and then sim a
// subset of them.
std::vector<SimmedPosition> select_slog_candidates(
  const std::vector<char>& buf, const Dictionary& dict, const SlogSimConfig& config,
  const std::vector<binlog::GamePositionIndex>& work, util::ProgressMeter* meter);

}  // namespace scribblez
