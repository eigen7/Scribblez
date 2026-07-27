#include "agent/endgame_hasty_bot.h"

#include "agent/agent_options.h"
#include "game/move.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace scribblez {

namespace {

// The per-thread EndgameSolver pool (see the class comment in the header):
// both seats of a thread's game loop draw the same instance.
std::shared_ptr<EndgameSolver> pooled_solver(int thread_id) {
  static std::mutex mutex;
  static std::map<int, std::shared_ptr<EndgameSolver>> pool;
  const std::lock_guard<std::mutex> lock(mutex);
  std::shared_ptr<EndgameSolver>& solver = pool[thread_id];
  if (!solver) solver = std::make_shared<EndgameSolver>();
  return solver;
}

}  // namespace

EndgameHastyBotAgent::EndgameHastyBotAgent(const Params& params)
    : HastyBotAgent(params.hasty),
      solver_params_(params.solver),
      solver_(pooled_solver(params.hasty.thread_id)) {
  solver_->clear();
}

MoveDecision EndgameHastyBotAgent::make_move(const MoveRequest& req) {
  // A proof certificate rides along as the decision's projection, so a
  // projection-respecting loop (self-play generation) fast-tracks the game to
  // its proven end instead of prompting for the remaining turns.
  if (req.bag_size == 0 && solver_params_.budget > 0) {
    const EndgameResult r = solver_->solve({&req.dict, req.board, req.my_rack, req.opp_rack,
                                            req.my_score, req.opp_score, scoreless_turns_},
                                           solver_params_);
    // Every solve the agent runs contributes to the operation totals, whether or
    // not its move ends up used.
    ++solve_totals_.solves;
    solve_totals_.nodes += r.nodes;
    solve_totals_.movegens += r.movegens;
    solve_totals_.certificate_nodes += r.certificate_nodes;
    solve_totals_.max_solve_nodes = std::max(solve_totals_.max_solve_nodes, r.nodes);
    // A proven-lost class-only result carries an arbitrary move (every move
    // loses, and that setting refines no further). With a certificate the
    // break-out takes priority -- the class-only setting exists to stop
    // spending compute on decided games, and it presumes a
    // projection-respecting loop. Without one, HastyBot's static-equity move
    // shapes the final spread better than an arbitrary losing move. With
    // spread_matters the solver spread-defends lost positions itself.
    const bool arbitrary_loss =
      !solver_params_.spread_matters && r.proven_class == -1 && r.continuation.empty();
    if (r.depth_completed >= 1 && !arbitrary_loss) return {r.best, r.continuation};
  }
  return HastyBotAgent::make_move(req);
}

void EndgameHastyBotAgent::observe_move(const Move& move) {
  if (move.type() == MoveType::PLAY)
    scoreless_turns_ = 0;
  else
    ++scoreless_turns_;
  HastyBotAgent::observe_move(move);
}

void EndgameHastyBotAgent::begin_game() {
  scoreless_turns_ = 0;
  solve_totals_ = {};
  solver_->clear();
  HastyBotAgent::begin_game();
}

std::unique_ptr<EndgameHastyBotAgent> EndgameHastyBotAgent::from_spec(
  const std::vector<std::string>& tokens, int thread_id, const std::string& name) {
  Params params;
  boost::program_options::options_description extra("hastybot-endgame options");
  params.solver.add_options(extra, "endgame-");
  params.hasty =
    HastyBotAgent::parse_hasty_params(tokens, thread_id, name, extra, "hastybot-endgame");
  return std::make_unique<EndgameHastyBotAgent>(params);
}

std::string EndgameHastyBotAgent::options_help() {
  EndgameSolver::Params defaults;  // scratch binding targets; only the defaults are read
  boost::program_options::options_description desc("hastybot-endgame options");
  defaults.add_options(desc, "endgame-");
  const std::string endgame = agent_options_help(
    "  HastyBot that solves the endgame once the bag empties: it plays HastyBot's\n"
    "  static-equity move while tiles remain, then hands the fully-known endgame\n"
    "  to an iterative-deepening negamax solver. Accepts every hastybot option\n"
    "  (listed below) in addition to these:\n",
    desc);
  return endgame + HastyBotAgent::options_help();
}

}  // namespace scribblez
