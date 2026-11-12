#pragma once

// For DeepBot's subtree dump (agent/subtree_dump.h): how often the policy
// head's top move at Bob's position is β(a, r), the reply his subtree chose
// (docs/plans/deep_bot.md, build order step 10's measure). The policy is asked
// the question from Bob's side, as PolicyBot would be at his turn: his rack,
// his belief about Alice's leave after `a` (bob_beliefs), and his pool of
// candidates cut as the model's pools were. Its top move over the whole pool
// is what the policy would play; over the candidates his subtree simmed, what
// it would pick among the moves the subtree could have chosen.

#include "agent/bob_subtree.h"
#include "agent/policy_scorer.h"
#include "agent/subtree_search.h"
#include "encoding/game_state_encoder.h"

#include <cstddef>
#include <vector>

namespace scribblez {

class Dictionary;

struct PolicyAgreement {
  double pool_share = 0;    // racks where the policy's top move over Bob's pool is β
  double simmed_share = 0;  // ... over the candidates his subtree simmed
};

// The agreement after Alice's candidate `c` of `turn`, over `replies`, Bob's
// reply at each of her racks for him. `after` is the game's encoder with the
// candidate applied, Bob to move; `params` are his subtree's, which sized the
// candidates it simmed, and `blank_prune_points` the model's pools' cut.
PolicyAgreement policy_agreement(const AliceTurn& turn, size_t c, const GameStateEncoder& after,
                                 const std::vector<BobReply>& replies, const BeliefModel& model,
                                 const SubtreeParams& params, int blank_prune_points,
                                 PolicyScorer& scorer, const Dictionary& dict);

}  // namespace scribblez
