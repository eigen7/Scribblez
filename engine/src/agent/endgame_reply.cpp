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
// bag, with Bob to move holding `bob`.
double endgame_value(const RolloutStart& applied, const Rack& bob, uint64_t budget,
                     const Dictionary& dict) {
  // The bag is empty, so the tiles Bob cannot see are her rack.
  const Rack alice = Rack::from_counts(applied.board.unseen_tiles(bob));
  SolverEndgameOracle oracle(thread_endgame_solver(), budget);
  const EndgameVerdict bobs = oracle.evaluate(
    {&dict, applied.board, bob, alice, applied.opp_score, applied.my_score, /*scoreless=*/0},
    EndgameSolver::Params{}.plies, EndgameGoal::kClass);
  const int cls =
    bobs.cls != EndgameVerdict::kClassUnknown ? bobs.cls : (bobs.spread > 0) - (bobs.spread < 0);
  return 0.5 * (1 - cls);
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
