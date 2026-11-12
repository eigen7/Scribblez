#include "agent/policy_agreement.h"

#include "agent/agent.h"
#include "sim/sim_runner.h"
#include "training/policy_set_inputs.h"

#include <algorithm>
#include <limits>

namespace scribblez {

namespace {

// The index of the largest of `logits` among `pool`'s moves that are in
// `among`, or pool.size() when none is.
size_t best_among(const std::vector<float>& logits, const std::vector<Move>& pool,
                  const std::vector<Move>& among) {
  size_t best = pool.size();
  for (size_t k = 0; k < pool.size(); ++k) {
    if (std::find(among.begin(), among.end(), pool[k]) == among.end()) continue;
    if (best == pool.size() || logits[k] > logits[best]) best = k;
  }
  return best;
}

}  // namespace

PolicyAgreement policy_agreement(const AliceTurn& turn, size_t c, const GameStateEncoder& after,
                                 const std::vector<BobReply>& replies, const BeliefModel& model,
                                 const SubtreeParams& params, int blank_prune_points,
                                 PolicyScorer& scorer, const Dictionary& dict) {
  const BobBeliefs beliefs =
    bob_beliefs(turn.position, turn.candidates[c], turn.bob_racks, model, dict);
  const Rack hidden;
  int pool_agree = 0;
  int simmed_agree = 0;
  for (size_t i = 0; i < turn.bob_racks.size(); ++i) {
    const SimPosition pos = mover_position(turn.applied[c], turn.bob_racks[i]);
    const int bag = pos.board.pov_bag_size(pos.rack.size());
    const MoveRequest req{pos.board, dict, pos.rack, hidden, pos.scores[0], pos.scores[1], bag};
    const std::vector<Move> pool =
      policy_set::pool_candidates(equity_top_k(req, std::numeric_limits<int>::max()),
                                  blank_prune_points)
        .moves;
    const auto belief = beliefs.sampler(i, *model.table);
    const std::vector<float>& logits = scorer.score(
      after, pos.rack, bag, policy_set::opp_rack_summary(pos.board, pos.rack, belief.get()), pool);
    const Move& chosen = replies[i].front().move;
    pool_agree += pool[best_logit(logits)] == chosen;
    const size_t simmed = best_among(logits, pool, mover_candidates(pos, params, dict));
    simmed_agree += simmed < pool.size() && pool[simmed] == chosen;
  }
  const double racks = double(turn.bob_racks.size());
  return {.pool_share = pool_agree / racks, .simmed_share = simmed_agree / racks};
}

}  // namespace scribblez
