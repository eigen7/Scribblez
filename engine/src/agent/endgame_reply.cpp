#include "agent/endgame_reply.h"

#include "agent/subtree_search.h"
#include "endgame/endgame_oracle.h"
#include "endgame/endgame_solver.h"
#include "game/rack.h"
#include "util/work_pool.h"

#include <algorithm>

namespace scribblez {

namespace {

// Alice's value of the endgame after `applied`, her move that emptied the
// bag, with Bob to move holding `bob`: the class the solver proves, or else
// the outcome when Bob plays the solve's move and both then play HastyBot's to
// the end. An unproven solve's move is usable and its value is not
// (docs/plans/deep_bot.md, "When Alice's move empties the bag"); a solve the
// budget declines has no move either, and HastyBot's stands in.
double endgame_value(const RolloutStart& applied, const Rack& bob, uint64_t budget,
                     const Dictionary& dict) {
  // The bag is empty, so the tiles Bob cannot see are her rack.
  const Rack alice = Rack::from_counts(applied.board.unseen_tiles(bob));
  const EndgameState state{
    &dict, applied.board, bob, alice, applied.opp_score, applied.my_score, /*scoreless_turns=*/0};
  const EndgameResult solved = thread_endgame_solver().solve(state, {.budget = budget});
  if (solved.proven_class != EndgameResult::kClassUnknown) return 0.5 * (1 - solved.proven_class);
  const int32_t bobs =
    greedy_playout_spread(state, solved.depth_completed >= 1 ? &solved.best : nullptr);
  return 0.5 * (1 - ((bobs > 0) - (bobs < 0)));
}

// The distinct racks of `racks`, and each rack's index among them: with a
// tile or two in the bag, a hundred sampled racks are a handful of racks.
struct DistinctRacks {
  std::vector<Rack> racks;
  std::vector<size_t> index;  // parallel to the input
};

DistinctRacks distinct(const std::vector<Rack>& racks) {
  DistinctRacks out;
  for (const Rack& r : racks) {
    const auto it = std::find(out.racks.begin(), out.racks.end(), r);
    out.index.push_back(size_t(it - out.racks.begin()));
    if (it == out.racks.end()) out.racks.push_back(r);
  }
  return out;
}

}  // namespace

std::vector<double> endgame_reply_values(const AliceTurn& turn, const std::vector<size_t>& which,
                                         uint64_t budget, const Dictionary& dict) {
  const DistinctRacks bob = distinct(turn.bob_racks);
  const size_t racks = bob.racks.size();
  std::vector<double> solved(which.size() * racks);
  util::parallel_for(solved.size(), [&](size_t k) {
    solved[k] = endgame_value(turn.applied[which[k / racks]], bob.racks[k % racks], budget, dict);
  });
  std::vector<double> out(turn.candidates.size(), kPruned);
  for (size_t j = 0; j < which.size(); ++j) {
    double sum = 0;
    for (const size_t r : bob.index) sum += solved[j * racks + r];
    out[which[j]] = sum / double(bob.index.size());
  }
  return out;
}

}  // namespace scribblez
