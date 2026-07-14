#include "agent/endgame_hasty_bot.h"

#include "agent/agent_options.h"
#include "game/move.h"

#include <boost/program_options.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace scribblez {

EndgameHastyBotAgent::EndgameHastyBotAgent(const Params& params)
    : HastyBotAgent(params.hasty),
      endgame_nodes_(params.endgame_nodes),
      endgame_plies_(params.endgame_plies),
      endgame_objective_(params.endgame_objective) {}

MoveDecision EndgameHastyBotAgent::make_move(const MoveRequest& req) {
  // Endgame: the bag is empty and both racks are fully known, so hand the
  // position to the exact solver (unless the solver is disabled). The solver's
  // move is used only when it completed at least its first iteration -- every
  // root move backed by a full greedy playout. Below that (the solve was
  // declined as too rich for the budget, or the budget ran out mid-iteration)
  // its answer reflects an arbitrary fraction of the root, and HastyBot's
  // static-equity argmax is the stronger policy.
  //
  // A solve that proved the game's class carries the proof-certificate line as
  // its continuation; it rides along as the decision's projection, so a
  // projection-respecting loop (self-play generation) fast-tracks the game to
  // its proven end instead of prompting for the remaining turns.
  if (req.bag_size == 0 && endgame_nodes_ > 0) {
    const EndgameResult r =
      solver_.solve(req.board, req.dict, req.my_rack, req.opp_rack, req.my_score, req.opp_score,
                    scoreless_turns_, endgame_nodes_, endgame_plies_, endgame_objective_);
    // A proven-lost first-win result carries an arbitrary move (every move
    // loses, and that objective refines no further). With a certificate the
    // break-out takes priority -- first-win exists to stop spending compute on
    // decided games, and it presumes a projection-respecting loop. Without
    // one, HastyBot's static-equity move shapes the final spread better than
    // an arbitrary losing move. The lexicographic objective spread-defends
    // lost positions itself.
    const bool arbitrary_loss = endgame_objective_ == EndgameObjective::kFirstWin &&
                                r.proven_class == -1 && r.continuation.empty();
    if (r.depth_completed >= 1 && !arbitrary_loss) return {r.best, r.continuation};
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

// The --player option values endgame_options() binds: from_spec() parses a
// token list into one, options_help() documents the same flags against its
// defaults.
struct EndgameOptionValues {
  uint64_t nodes = EndgameHastyBotAgent::kDefaultEndgameNodes;
  int plies = 25;
  std::string objective = endgame_objective_name(EndgameObjective::kLexicographic);
};

boost::program_options::options_description endgame_options(EndgameOptionValues& values) {
  namespace po = boost::program_options;
  po::options_description desc("hastybot-endgame options");
  desc.add_options()  //
    ("endgame-nodes", po::value<uint64_t>(&values.nodes)->default_value(values.nodes),
     "per-turn endgame-solver node budget (0 disables the solver)")  //
    ("endgame-plies", po::value<int>(&values.plies)->default_value(values.plies),
     "endgame-solver iterative-deepening depth cap")  //
    ("endgame-objective",
     po::value<std::string>(&values.objective)->default_value(values.objective),
     "what the solver optimizes: lexicographic (prove win/draw/loss, then maximize "
     "spread without trading the class for points), first-win (class proof only; the "
     "self-play break-out objective), or spread (pure margin)");
  return desc;
}

}  // namespace

std::unique_ptr<EndgameHastyBotAgent> EndgameHastyBotAgent::from_spec(
  const std::vector<std::string>& tokens, int thread_id, const std::string& name) {
  EndgameOptionValues values;
  boost::program_options::options_description extra = endgame_options(values);

  Params params;
  params.hasty =
    HastyBotAgent::parse_hasty_params(tokens, thread_id, name, extra, "hastybot-endgame");
  params.endgame_nodes = values.nodes;
  params.endgame_plies = values.plies;
  params.endgame_objective = parse_endgame_objective(values.objective);
  return std::make_unique<EndgameHastyBotAgent>(params);
}

std::string EndgameHastyBotAgent::options_help() {
  EndgameOptionValues values;  // scratch binding targets; only the defaults are read
  const std::string endgame = agent_options_help(
    "  HastyBot that solves the endgame once the bag empties: it plays HastyBot's\n"
    "  static-equity move while tiles remain, then hands the fully-known endgame\n"
    "  to an iterative-deepening negamax solver. Accepts every hastybot option\n"
    "  (listed below) in addition to these:\n",
    endgame_options(values));
  return endgame + HastyBotAgent::options_help();
}

}  // namespace scribblez
