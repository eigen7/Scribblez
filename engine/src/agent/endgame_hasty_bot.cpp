#include "agent/endgame_hasty_bot.h"

#include "agent/agent_options.h"
#include "game/move.h"

#include <boost/program_options.hpp>

#include <memory>
#include <optional>
#include <string>

namespace scribblez {

EndgameHastyBotAgent::EndgameHastyBotAgent(const Params& params)
    : HastyBotAgent(params.hasty), endgame_(params.hasty.thread_id, params.solver) {}

MoveDecision EndgameHastyBotAgent::make_move(const MoveRequest& req) {
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;
  return HastyBotAgent::make_move(req);
}

void EndgameHastyBotAgent::observe_move(const Move& move) {
  endgame_.observe_move(move);
  HastyBotAgent::observe_move(move);
}

void EndgameHastyBotAgent::begin_game(const BeginGameRequest& req) {
  endgame_.begin_game();
  HastyBotAgent::begin_game(req);
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
