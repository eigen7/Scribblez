#pragma once

// Monte-Carlo sims over sampled positions of a loaded .slog file: replay each
// position to its pre-move decision point, rank the legal candidates by
// HastyBot static equity, select some, and run SimRunner over them (common
// random numbers). The one implementation behind every tool that sims .slog
// positions with a model-free candidate recipe -- sim_obs_tool's .sobs
// sidecars and the sim-labeled-candidates measurement and labeler
// (docs/plans/sim_labeled_candidates.md) -- so they cannot drift on the replay,
// the information condition, or the seeding.

#include "data/slog_sampling.h"
#include "game/move.h"
#include "sim/rollout_summary.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_candidates.h"
#include "util/progress.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace scribblez {

class Dictionary;

// Which of a position's equity-ranked legal moves get simmed: the stratified
// sample (move_set_eval::stratified_candidates) when `quotas` is set, else the
// flat top_k prefix of the ranking.
struct SimCandidateRecipe {
  int top_k = 10;
  std::optional<move_set_eval::StratumQuotas> quotas;
};

// A position's selected candidates, each with its 0-based rank in the
// position's static-equity ranking (-1 for a played move the generator never
// enumerates: a PASS chosen while other moves were legal). Empty `moves` means
// the selector declines the position: it is not simmed.
struct SimCandidates {
  std::vector<Move> moves;
  std::vector<int32_t> equity_ranks;
  // Parallel to `moves`: the candidates a selector is asking about, when it has
  // such a notion (the setup selector's setup plays); all false otherwise.
  std::vector<char> highlighted;
  // Parallel to `moves`: HastyBot static equity. Filled by the simmer, not by
  // selectors.
  std::vector<double> equities;
  uint32_t num_legal_moves = 0;
};

// Apply `recipe` to `ranked` (every legal move, best equity first). `played`
// is the move the game made at the position, which the stratified sample
// stores first; `rng` drives the sampled strata.
SimCandidates select_sim_candidates(const std::vector<Move>& ranked, const Move& played,
                                    const SimCandidateRecipe& recipe, std::mt19937_64& rng);

// Chooses a position's candidates from `ranked` (every legal move, best equity
// first). `played` is the move the game made there; `rng` is seeded per
// position, so a selector is deterministic in (run seed, game, turn).
using SimCandidateSelector =
  std::function<SimCandidates(const SimPosition& pos, const std::vector<Move>& ranked,
                              const Move& played, std::mt19937_64& rng)>;

SimCandidateSelector recipe_selector(const SimCandidateRecipe& recipe);

// Positions that allow a high-value setup (sim/setup_plays.h): the top `cut` of
// the ranking plus every setup play, the latter highlighted and capped at the
// `max_setups` best-ranked. Declines a position with no setup play outside the
// cut -- there the cut hides nothing.
SimCandidateSelector setup_selector(const Dictionary& dict, int cut, int max_setups);

// Every position: the top `cut` of the ranking plus every play that places no
// blank (the blank-placing plays are most of a blank rack's thousands of legal
// moves, nearly all of them designation variants of each other), capped at the
// `max_plays` best-ranked beyond the cut (0 = no cap). High-value setups are
// highlighted. Both this and setup_selector list the cut first, in rank order.
SimCandidateSelector all_plays_selector(const Dictionary& dict, int cut, int max_plays);

struct SlogSimConfig {
  // Sim with the opponent's retained leave known (the open-leaves information
  // condition); the candidate ranking then sees it too.
  bool open_leaves = false;
  SimCandidateSelector selector = recipe_selector({});
  // Per-position runner params. Parallelism is across positions, so `threads`
  // here must be 1: that utilizes cores better than within-position threading
  // and keeps every position's sims independent of the worker count.
  SimRunner::Params runner;
  uint64_t seed = 0;  // run seed; each position sims under position_seed(seed, game, turn)
  // Added to each position's SimRunner seed and to nothing else, so runs that
  // differ only here sim the same positions and candidates with independent
  // rollouts (rollout i draws seed base + i, so offsets a rollout count apart
  // do not overlap) -- what a held-out estimate of a sim pick's value needs.
  uint64_t rollout_seed_offset = 0;
  int threads = 1;  // position workers
  // What each position's rollouts are reduced to. Observations are the training
  // currency (35 KB a candidate); summaries are the analysis one
  // (sim/rollout_summary.h), cheap enough to keep for every legal play.
  bool keep_observations = true;
  bool keep_summaries = false;
  // With summaries: the first this-many candidates are references, and every
  // candidate gets its paired win difference against each of them.
  int paired_references = 0;
};

struct SimmedPosition {
  binlog::GamePositionIndex pos;
  SimPosition position;    // the replayed decision point
  uint64_t base_seed = 0;  // the SimRunner::run seed used
  SimCandidates candidates;
  int bag_size = 0;  // tiles in the bag at the decision point
  Move played;       // the move the game made here
  // Parallel to candidates.moves; each filled per SlogSimConfig.
  std::vector<SimObservation> observations;
  std::vector<RolloutSummary> summaries;
  // paired[c][r]: candidate c's win value minus reference r's, over the rollouts.
  std::vector<std::vector<PairedWinDiff>> paired;
};

// Sim every position of `work` within the loaded .slog bytes `buf`, returning
// results in `work` order (so a sorted work list gives output that is
// byte-stable across thread counts). `meter` advances once per position.
// A failure inside a worker rethrows here as util::Exception naming the
// position ("game G turn T: ...").
// Declined positions come back with empty candidates and observations.
std::vector<SimmedPosition> sim_slog_positions(const std::vector<char>& buf, const Dictionary& dict,
                                               const SlogSimConfig& config,
                                               const std::vector<binlog::GamePositionIndex>& work,
                                               util::ProgressMeter* meter);

// The selection half alone: every position's candidates, no sims (observations
// stay empty). Selection is deterministic, so a caller can scan a file for the
// positions a selector accepts and then sim a subset of them.
std::vector<SimmedPosition> select_slog_candidates(
  const std::vector<char>& buf, const Dictionary& dict, const SlogSimConfig& config,
  const std::vector<binlog::GamePositionIndex>& work, util::ProgressMeter* meter);

}  // namespace scribblez
