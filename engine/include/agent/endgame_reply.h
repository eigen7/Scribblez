#pragma once

// DeepBot's `subtree` reply model where Alice's candidate empties the bag
// (docs/plans/deep_bot.md, "When Alice's move empties the bag"). Bob then
// holds every tile he cannot see but hers, so he knows her rack: his reply is
// an endgame decision, and his subtree, a search under uncertainty, does not
// apply. The endgame solver makes it instead, and Alice's value at each rack
// she sampled for him is the endgame's value.
//
// The hasty sims of such a candidate play both sides' HastyBot moves to the
// game's end, a weaker Bob than the one who answers her other candidates from
// his subtree; valued that way beside them, the candidates that empty the bag
// read too well.

#include "agent/bob_subtree.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

// Alice's value of each candidate c in `which`, each of which empties the
// bag: the mean over her racks for Bob of the endgame after it, 1, 1/2 or 0
// for her by the class the solver proves within `budget` nodes, or unproven
// by the sign of its spread estimate. kPruned for the other candidates.
std::vector<double> endgame_reply_values(const AliceTurn& turn, const std::vector<size_t>& which,
                                         uint64_t budget, const Dictionary& dict);

}  // namespace scribblez
