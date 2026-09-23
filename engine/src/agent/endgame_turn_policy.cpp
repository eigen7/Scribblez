#include "agent/endgame_turn_policy.h"

#include "game/move.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace scribblez {

namespace {

// Both seats of a game thread draw the same instance.
std::shared_ptr<EndgameSolver> pooled_solver(int thread_id) {
  static std::mutex mutex;
  static std::map<int, std::shared_ptr<EndgameSolver>> pool;
  const std::lock_guard<std::mutex> lock(mutex);
  std::shared_ptr<EndgameSolver>& solver = pool[thread_id];
  if (!solver) solver = std::make_shared<EndgameSolver>();
  return solver;
}

}  // namespace

EndgameTurnPolicy::EndgameTurnPolicy(int thread_id, const EndgameSolver::Params& params)
    : params_(params), solver_(pooled_solver(thread_id)) {
  solver_->clear();
}

std::optional<MoveDecision> EndgameTurnPolicy::try_solve(const MoveRequest& req) {
  if (req.bag_size != 0 || params_.budget == 0) return std::nullopt;

  const auto t0 = std::chrono::steady_clock::now();
  const EndgameResult r = solver_->solve({&req.dict, req.board, req.my_rack, req.opp_rack,
                                          req.my_score, req.opp_score, scoreless_turns_},
                                         params_);
  // Every solve counts toward the totals, whether or not its move is used.
  solve_totals_.solve_ns += uint64_t(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
      .count());
  ++solve_totals_.solves;
  solve_totals_.nodes += r.nodes;
  solve_totals_.movegens += r.movegens;
  solve_totals_.certificate_nodes += r.certificate_nodes;
  solve_totals_.max_solve_nodes = std::max(solve_totals_.max_solve_nodes, r.nodes);

  // A proven loss without spread_matters carries an arbitrary move: every move
  // loses, and the class-only solve refines no further. If it comes with a
  // certificate, play it anyway so the game loop can end the decided game.
  // Without one, the owning agent's move defends the spread better than an
  // arbitrary losing move. (With spread_matters the solver itself maximizes
  // the spread of a lost position.)
  const bool arbitrary_loss =
    !params_.spread_matters && r.proven_class == -1 && r.continuation.empty();
  if (r.depth_completed < 1 || arbitrary_loss) return std::nullopt;
  return MoveDecision{r.best, r.continuation};
}

void EndgameTurnPolicy::observe_move(const Move& move) {
  if (move.type() == MoveType::PLAY)
    scoreless_turns_ = 0;
  else
    ++scoreless_turns_;
}

void EndgameTurnPolicy::begin_game() {
  scoreless_turns_ = 0;
  solve_totals_ = {};
  solver_->clear();
}

}  // namespace scribblez
