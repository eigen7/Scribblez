#pragma once

// DeepBot's `subtree` reply model (docs/plans/deep_bot.md, "Bob's subtree"):
// for each of Alice's candidates `a` and each rack `r` she sampled for Bob,
// Bob's reply β(a, r) as he would choose it on his own information, and
// Alice's value of `a` given those replies.
//
// Bob samples Alice's leave from his belief (belief::ReplyView: the leave
// inference row of his position, Alice's private past hidden), sims his
// static-equity top moves on those samples (Alice and Bob reply by HastyBot,
// the leaf network at the end), and plays the one most likely best. Only the
// move crosses back: Alice then sims each (a, r) after β(a, r) with her own
// leave known and her replenishment drawn per sim, and values `a` by the mean.
//
// Every subtree of a turn races through one PoolLabeler, so their rounds share
// launches; so do Alice's sims. Jobs are built a candidate at a time, in
// parallel, as the labeler asks for them, so a turn never holds them all.

#include "belief/leave_params.h"
#include "belief/reply_query.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"

#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

struct SubtreeParams {
  int alice_racks = 20;      // R_B: Alice's leaves Bob samples, one rollout each
  int candidates = 8;        // N_B, Bob's, by static equity
  int late_candidates = 24;  // ... with late_bag or fewer tiles in the bag
  int late_bag = 40;
  bool race = true;  // stop Bob's clearly beaten candidates between rounds
};

// Bob's reply at one (candidate, rack), with what the diagnostics read.
struct BobReply {
  Move move;
  int equity_rank = 0;  // among Bob's candidates; 0 is HastyBot's move
  int candidates = 1;   // how many Bob simmed (1: no sims)
  double q = 1;         // the probability his sims gave that it is his best
};

// What the subtrees read of Alice's turn. `applied` is parallel to
// `candidates`: each applied to the board as its sims' start
// (make_rollout_start). Rack i of `bob_racks` is the rack Alice's rollout i
// seats for Bob.
struct AliceTurn {
  const belief::ReplyPosition& position;
  const std::vector<Move>& candidates;
  const std::vector<RolloutStart>& applied;
  const std::vector<Rack>& bob_racks;
};

// The leave inference model Bob's beliefs come from; an empty `predict` draws
// Alice's whole rack uniformly.
struct BeliefModel {
  belief::LeaveParamsFn predict;
  const LeaveValueArrays* table = nullptr;
};

// replies[c][i] = β(candidates[c], bob_racks[i]), for each c in `which`; the
// other rows stay empty. `seed` keys Bob's samples and draws, by rack index.
std::vector<std::vector<BobReply>> bob_replies(const AliceTurn& turn,
                                               const std::vector<size_t>& which,
                                               const BeliefModel& model, RolloutValues& values,
                                               const SubtreeParams& params, const Dictionary& dict,
                                               uint64_t seed);

// Alice's value of each candidate c in `which` from her sims after Bob's
// replies: the mean over racks i of her win probability (a draw counting
// half) over `sims` rollouts after β(c, i), each drawing her replenishment.
// Other entries are 0.
std::vector<double> values_after_replies(const AliceTurn& turn, const std::vector<size_t>& which,
                                         const std::vector<std::vector<BobReply>>& replies,
                                         RolloutValues& values, int sims, const Dictionary& dict,
                                         uint64_t seed);

// Alice's full rack in one of her sims after Bob's reply, Bob holding
// `bob_rack` after her candidate `applied`: her leave and her replenishment,
// drawn from the bag as she found it. That bag holds neither her leave nor the
// tiles of an exchange, which went back only after she drew; Bob's draw after
// his reply, made by the sim, can take them.
Rack alice_rack_after_reply(const RolloutStart& applied, const Rack& bob_rack, uint64_t seed);

// The racks Alice's rollouts seat for Bob: rollout i's for i < count, from the
// group her candidates share (any of `starts`, all in `group`).
std::vector<Rack> bob_racks(const RolloutGroup& group, const RolloutStart& start, int count);

}  // namespace scribblez
