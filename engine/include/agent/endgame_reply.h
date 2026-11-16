#pragma once

// Bob's reply where Alice's candidate empties the bag (docs/plans/deep_bot.md,
// "When Alice's move empties the bag"), under DeepBot's `subtree` and `hasty`
// reply models. Bob then holds every tile he cannot see but hers, so he knows
// her rack: his reply is an endgame decision, which neither his subtree, a
// search under uncertainty, nor HastyBot's rule makes well. The endgame
// solver makes it instead, and Alice's value at each rack she sampled for him
// is the endgame's value.
//
// The hasty sims of such a candidate play both sides' HastyBot moves to the
// game's end: a Bob who walks past the fish or the setup that one who sees
// her rack answers, so the candidates that empty the bag read too well.

#include "agent/subtree_search.h"
#include "game/rack.h"
#include "sim/hasty_rollout_task.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

// Alice's value of each candidate c in `which`, each of which empties the
// bag and leaves the position `applied[c]`: the mean over `bob_racks`, her
// racks for Bob, of the endgame after it, 1, 1/2 or 0 for her by the class
// the solver proves within `budget` nodes, or unproven by the outcome when
// Bob plays the solve's move and both then play HastyBot's to the end.
// kPruned for the other candidates.
std::vector<double> endgame_reply_values(const std::vector<RolloutStart>& applied,
                                         const std::vector<Rack>& bob_racks,
                                         const std::vector<size_t>& which, uint64_t budget,
                                         const Dictionary& dict);

}  // namespace scribblez
