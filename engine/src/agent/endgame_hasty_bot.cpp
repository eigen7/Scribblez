#include "agent/endgame_hasty_bot.h"

#include "agent/agent_options.h"
#include "game/move.h"

#include <boost/program_options.hpp>

#include <string>
#include <utility>

namespace scribblez {

namespace {

// True if `m` is a move the agent may legally play in `req`: a pass (always legal
// in the endgame, where the bag is empty and exchanging is not) or one of the
// generated legal plays. The solver returns its moves from exactly this set, so
// this only guards against a solver defect -- a safety net, not an expected path.
bool solver_move_playable(const Move& m, const MoveRequest& req) {
  if (m.type() == MoveType::PASS) return true;
  for (const Move& legal : generate_legal_plays(req))
    if (m == legal) return true;
  return false;
}

}  // namespace

EndgameHastyBotAgent::EndgameHastyBotAgent(const Params& params)
    : HastyBotAgent(params.hasty),
      endgame_nodes_(params.endgame_nodes),
      endgame_plies_(params.endgame_plies) {}

Move EndgameHastyBotAgent::make_move(const MoveRequest& req) {
  // Endgame: the bag is empty and both racks are fully known, so hand the
  // position to the exact solver (unless the solver is disabled). If the solver
  // ever hands back a move that is not legal here, fall through to HastyBot.
  if (req.bag_size == 0 && endgame_nodes_ > 0) {
    const EndgameResult r =
      solver_.solve(req.board, req.dict, req.my_rack, req.opp_rack, req.my_score, req.opp_score,
                    scoreless_turns_, endgame_nodes_, endgame_plies_);
    if (solver_move_playable(r.best, req)) return r.best;
  }
  return HastyBotAgent::make_move(req);
}

void EndgameHastyBotAgent::observe_move(const Move& move) {
  // Track consecutive scoreless turns for the solver's scoreless-turn input,
  // mirroring the game loop: any play resets the run, a pass or exchange
  // extends it.
  if (move.type() == MoveType::PLAY)
    scoreless_turns_ = 0;
  else
    ++scoreless_turns_;
  HastyBotAgent::observe_move(move);
}

void EndgameHastyBotAgent::begin_game() {
  scoreless_turns_ = 0;
  solver_.clear();
  HastyBotAgent::begin_game();
}

namespace {

// The endgame-solver --player options, binding the two knobs to the given
// storage. from_spec() builds this to extend the HastyBot option parse;
// options_help() builds it against scratch defaults to document the same flags.
boost::program_options::options_description endgame_options(uint64_t& endgame_nodes,
                                                            int& endgame_plies) {
  namespace po = boost::program_options;
  po::options_description desc("hastybot-endgame options");
  desc.add_options()  //
    ("endgame-nodes", po::value<uint64_t>(&endgame_nodes)->default_value(endgame_nodes),
     "per-turn endgame-solver node budget (0 disables the solver)")  //
    ("endgame-plies", po::value<int>(&endgame_plies)->default_value(endgame_plies),
     "endgame-solver iterative-deepening depth cap");
  return desc;
}

}  // namespace

std::unique_ptr<EndgameHastyBotAgent> EndgameHastyBotAgent::from_spec(
  const std::vector<std::string>& tokens, int thread_id, const std::string& name) {
  uint64_t endgame_nodes = 50000;
  int endgame_plies = 25;
  boost::program_options::options_description extra = endgame_options(endgame_nodes, endgame_plies);

  Params params;
  params.hasty =
    HastyBotAgent::parse_hasty_params(tokens, thread_id, name, extra, "hastybot-endgame");
  params.endgame_nodes = endgame_nodes;
  params.endgame_plies = endgame_plies;
  return std::make_unique<EndgameHastyBotAgent>(params);
}

std::string EndgameHastyBotAgent::options_help() {
  uint64_t endgame_nodes = 50000;
  int endgame_plies = 25;  // scratch binding targets; never read here
  const std::string endgame = agent_options_help(
    "  HastyBot that solves the endgame once the bag empties: it plays HastyBot's\n"
    "  static-equity move while tiles remain, then hands the fully-known endgame\n"
    "  to an iterative-deepening negamax solver. Accepts every hastybot option\n"
    "  (listed below) in addition to these:\n",
    endgame_options(endgame_nodes, endgame_plies));
  return endgame + HastyBotAgent::options_help();
}

}  // namespace scribblez
