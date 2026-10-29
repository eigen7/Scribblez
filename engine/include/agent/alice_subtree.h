#pragma once

// DeepBot's `alice-subtree` reply model, for a HastyBot opponent: for each of
// Alice's candidates `a` and each rack `r` she sampled for Bob, Bob replies by
// HastyBot, which against that opponent is his move exactly; Alice then
// chooses her second move α(a, r) by her own sims on her own information, and
// two hasty plies after it, the leaf network at the end, value `a`. It is Bob's
// subtree (agent/bob_subtree.h) with the search moved to Alice's second turn,
// so the opponent model is exact and what is left to measure is Alice
// modelling her own later play as a searcher rather than as HastyBot.
//
// At her second turn Alice holds the rack rollout r dealt her and sees Bob's
// reply, not his rack: she samples his leave from her belief after the reply
// (belief::ReplyView with Bob as the chooser), or his whole rack uniformly
// without a model, and sims her static-equity top moves (Bob and Alice reply
// by HastyBot, the leaf network at the end). Only her move crosses back: the
// sims after it seat Bob's leave as rack r left it and draw both players'
// replenishment.

#include "agent/bob_subtree.h"
#include "agent/subtree_search.h"
#include "belief/reply_query.h"
#include "game/move.h"
#include "game/rack.h"
#include "sim/hasty_rollout_task.h"
#include "sim/rollout_values.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace scribblez {

class Dictionary;

// Rollout i of one of Alice's candidates, dealt, up to her second turn.
struct SecondTurn {
  Rack bob_rack;     // before his reply
  int bag = 0;       // the bag he replied with, before his draw
  Move reply;        // HastyBot's
  RolloutStart bob;  // the reply applied, from his side: his leave, its exchanged tiles
  Rack alice_rack;   // after her refill
};

// The second turn of each rack of `turn.bob_racks` after candidate `c`, which
// must leave tiles in the bag.
std::vector<SecondTurn> second_turns(const AliceTurn& turn, size_t c, const Dictionary& dict);

// Alice's value of each candidate c in `which`: the mean over racks i of her
// win probability (a draw counting half) over `sims` rollouts after the move
// her sims chose at second_turns(c)[i]. `bob_earlier` is Bob's moves before
// his reply, which her belief's history reads. Other entries are kPruned.
std::vector<double> alice_subtree_values(const AliceTurn& turn, const std::vector<size_t>& which,
                                         std::span<const belief::ChooserMove> bob_earlier,
                                         const BeliefModel& model, RolloutValues& values,
                                         const SubtreeParams& params, int sims,
                                         const Dictionary& dict, uint64_t seed);

}  // namespace scribblez
