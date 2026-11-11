#pragma once

// DeepBot's `subtree` reply model (docs/plans/deep_bot.md, "Bob's subtree"):
// for each of Alice's candidates `a` and each rack `r` she sampled for Bob,
// Bob's reply β(a, r) as he would choose it on his own information, and
// Alice's value of `a` given those replies.
//
// Bob samples Alice's leave from his belief (belief::ReplyView: the leave
// inference row of his position, Alice's private past hidden), sims his
// static-equity top moves on those samples (Alice and Bob reply by HastyBot,
// the leaf network at the end), and his reply is his moves with their chances
// of being his best. At R_B racks his sims separate close moves poorly, so a
// near-tie does not snap to one move: Alice's sims at the rack are apportioned
// among his moves by those chances (contenders()), and the moves that get
// any are his contenders. Only the moves cross back: Alice sims each (a, r)
// after each contender with her own leave known and her replenishment drawn
// per sim, and values `a` by the mean over racks of her payoffs weighted by
// the contenders' chances.
//
// Every subtree of a turn races through one PoolLabeler, so their rounds share
// launches; so do Alice's sims. Jobs are built a candidate at a time, in
// parallel (agent/subtree_search.h).

#include "agent/subtree_search.h"
#include "belief/leave_params.h"
#include "belief/reply_query.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/hasty_rollout_task.h"
#include "sim/map_residuals.h"
#include "sim/rollout_values.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

// What the subtrees read of Alice's turn. `applied` is parallel to
// `candidates`: each applied to the board as its sims' start
// (make_rollout_start), all in `group`. Rack i of `bob_racks` is the rack
// Alice's rollout i seats for Bob, the group's rollout i dealing it.
struct AliceTurn {
  const belief::ReplyPosition& position;
  const std::vector<Move>& candidates;
  const std::vector<RolloutStart>& applied;
  const std::vector<Rack>& bob_racks;
  const RolloutGroup& group;
};

// Bob's reply at one rack, as Alice's sims use it: his contenders, the
// likeliest first, so the front is β(a, r); each one's weight is its share
// of her value at the rack.
using BobReply = std::vector<SubtreeMove>;

// replies[c][i] = Bob's reply to candidates[c] at bob_racks[i], for each c in
// `which`; the other rows stay empty. `sims` counts the rollouts run.
struct BobReplies {
  std::vector<std::vector<BobReply>> replies;
  long long sims = 0;
};

// `sims` is Alice's sims after his reply at each rack, which his contenders
// share; `seed` keys his samples and draws, by rack index.
BobReplies bob_replies(const AliceTurn& turn, const std::vector<size_t>& which,
                       const BeliefModel& model, RolloutValues& values, const SubtreeParams& params,
                       int sims, const Dictionary& dict, uint64_t seed);

// Alice's sims after Bob's replies to each candidate c in `which`.
struct ReplyPayoffs {
  // payoffs[c][i][k]: Alice's win probability (a draw counting half) over
  // the rollouts after contender k of replies[c][i], its `sims` of them, each
  // drawing her replenishment; one rack's contenders share draws, rollout j
  // of each dealing the same tiles. The other candidates' rows stay empty.
  std::vector<std::vector<std::vector<double>>> payoffs;
  // Alice's value of c: the mean over racks of her payoffs weighted by the
  // contenders' weights; kPruned for the rest.
  std::vector<double> values;
  // residuals[c][i][k], parallel to payoffs when asked for: the sims after
  // the contender against the leaf network's prior on the position after it,
  // from Bob's side (sim/map_residuals.h).
  std::vector<std::vector<std::vector<MapResidual>>> residuals;
  long long sims = 0;  // the rollouts run
};

// With `residuals`, the sims sum their maps and each (c, i, k) gets its
// residual summary: one leaf network call per contender more.
ReplyPayoffs payoffs_after_replies(const AliceTurn& turn, const std::vector<size_t>& which,
                                   const std::vector<std::vector<BobReply>>& replies,
                                   RolloutValues& values, const Dictionary& dict, uint64_t seed,
                                   bool residuals = false);

// The racks Alice's rollouts seat for Bob: rollout i's for i < count, from the
// group her candidates share (any of `starts`, all in `group`).
std::vector<Rack> bob_racks(const RolloutGroup& group, const RolloutStart& start, int count);

}  // namespace scribblez
