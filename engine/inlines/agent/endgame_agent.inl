#include "agent/endgame_agent.h"

#include "agent/agent_options.h"
#include "game/move.h"

#include <boost/program_options.hpp>

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace scribblez {

template <class Base>
EndgameAgent<Base>::EndgameAgent(const Params& params)
    : Base(params.base), endgame_(params.base.thread_id, params.solver) {}

template <class Base>
MoveDecision EndgameAgent<Base>::make_move(const MoveRequest& req) {
  if (const std::optional<MoveDecision> solved = endgame_.try_solve(req)) return *solved;
  return Base::make_move(req);
}

template <class Base>
void EndgameAgent<Base>::observe_move(const Move& move) {
  endgame_.observe_move(move);
  Base::observe_move(move);
}

template <class Base>
void EndgameAgent<Base>::begin_game(const BeginGameRequest& req) {
  endgame_.begin_game();
  Base::begin_game(req);
}

template <class Base>
std::unique_ptr<EndgameAgent<Base>> EndgameAgent<Base>::from_spec(
  const std::vector<std::string>& tokens, int thread_id, const std::string& name) {
  const std::string type = std::string(Base::kType) + "-endgame";
  Params params;
  boost::program_options::options_description extra(type + " options");
  params.solver.add_options(extra, "endgame-");
  params.base = Base::parse_params(tokens, thread_id, name, extra, type.c_str());
  return std::make_unique<EndgameAgent>(params);
}

template <class Base>
std::string EndgameAgent<Base>::options_help() {
  const std::string_view type = Base::kType;
  EndgameSolver::Params defaults;  // scratch binding targets; only the defaults are read
  boost::program_options::options_description desc(std::format("{}-endgame options", type));
  defaults.add_options(desc, "endgame-");
  const std::string description = std::format(
    "  {0} that solves the endgame once the bag empties: it plays exactly as\n"
    "  --type={0} while tiles remain, then hands the fully-known endgame to an\n"
    "  iterative-deepening negamax solver. Accepts every {0} option (listed\n"
    "  below) in addition to these:\n",
    type);
  return agent_options_help(description, desc) + Base::options_help();
}

}  // namespace scribblez
