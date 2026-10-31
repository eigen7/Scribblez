#pragma once

// What the policy-set model reads of a position (docs/plans/deep_bot.md, "The
// policy-set model"), beyond the board row: the candidates it chooses among,
// the chooser's belief about the opponent's rack, and per candidate the move
// set model's move features plus its static equity. Training (the pool labeler
// and the dataset's FFI replay) and PolicyBot at play build every one of these
// here, so the two cannot drift. The format version (policy_set_format.h)
// covers the opponent-rack and equity inputs.

#include "belief/leave_sampler.h"
#include "belief/opp_rack_summary.h"
#include "encoding/game_state_encoder.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "training/move_set_encoder.h"

#include <array>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

namespace policy_set {

// The opponent-rack input: the 27 presence probabilities, whether the leave
// was modelled, and the leave value's mean and sd in units of kValueScale
// points.
inline constexpr int kOppRackFloats = TILE_KINDS + 3;
inline constexpr float kValueScale = 10.0f;
// The equity input: a candidate's static equity less the best of its
// position's, in kValueScale points, floored kEquityFloor below (100 points).
inline constexpr float kEquityFloor = -10.0f;

// A position's pool: the candidates the model chooses among, and each one's
// 0-based rank in the static-equity ranking (-1 for a pass outside it).
struct PoolCandidates {
  std::vector<Move> moves;
  std::vector<int32_t> equity_ranks;
};

// The pool of `ranked`, every legal move best static equity first: every legal
// play, every distinct exchange, and the pass, minus the blank moves (a play
// that places the blank, or an exchange that returns it) that score
// `blank_prune_points` or more below the greedy move, the highest-scoring play.
// An exchange scores 0 under this rule. A blank multiplies the moves through
// its designations; the prune keeps the ones that could be best.
PoolCandidates pool_candidates(const std::vector<Move>& ranked, int blank_prune_points);

// The chooser's belief about the opponent's rack at a position where they hold
// `rack` on `board`: `leave` (null: none modelled) plus a uniform refill from
// the tiles they cannot see.
belief::OppRackSummary opp_rack_summary(const Board& board, const Rack& rack,
                                        const belief::LeaveSampler* leave);

// The opponent-rack input of `summary`.
std::array<float, kOppRackFloats> encode_opp_rack(const belief::OppRackSummary& summary);

// One position's candidates as the model reads them: the move set model's
// features, and each candidate's equity input.
struct CandidateInputs {
  move_set::MoveFeatureArrays features;
  std::vector<float> equity;

  int count() const { return features.count; }

  // Encode `moves` at the decision point `position` replays to, its active
  // player to move holding `rack` with `bag_size` tiles in the bag. The equity
  // is HastyBot's with the opponent's leave hidden.
  void encode(const GameStateEncoder& position, const Dictionary& dict, const Rack& rack,
              int bag_size, const std::vector<Move>& moves);
};

}  // namespace policy_set
}  // namespace scribblez
