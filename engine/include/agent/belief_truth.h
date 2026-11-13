#pragma once

// For DeepBot's subtree dump (agent/subtree_dump.h): what Bob's belief about
// Alice's leave makes of the leave she in fact keeps after one of her
// candidates. In his subtrees Bob draws her leave from the leave inference
// model run from his side (bob_beliefs), a model of how the policy that
// played its training games chooses moves. A candidate that policy would not
// play is outside what the model learned, and the more his belief misses her
// real leave there, the less his subtree's reply says about how a real
// opponent would answer her: a setup that keeps a tile for a hook looks, to
// a model of an equity maximizer, like a rack without the tile.

#include "agent/bob_subtree.h"
#include "agent/subtree_search.h"

#include <cstddef>

namespace scribblez {

class Dictionary;

struct BeliefTruth {
  int racks = 0;  // of Bob's, those whose belief the model gave
  // The mean over them of log p(the leave Alice keeps): under the model's
  // belief, and under no information (a uniform draw from his unseen pool).
  double log_prob = 0;
  double prior_log_prob = 0;
};

// The belief's standing after Alice's candidate `c` of `turn`, over her racks
// for Bob. All zero without a model, or where it has nothing to say (she
// keeps no tile, or passes).
BeliefTruth belief_truth(const AliceTurn& turn, size_t c, const BeliefModel& model,
                         const Dictionary& dict);

}  // namespace scribblez
