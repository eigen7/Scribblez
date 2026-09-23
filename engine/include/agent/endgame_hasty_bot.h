#pragma once

#include "agent/endgame_turn_policy.h"
#include "agent/hasty_bot.h"
#include "endgame/endgame_solver.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// HastyBot with an exact endgame: bag-empty turns go to an EndgameTurnPolicy,
// and every other turn, like any the solver declines, plays HastyBot's move.
class EndgameHastyBotAgent : public HastyBotAgent {
 public:
  struct Params {
    HastyBotAgent::Params hasty;
    EndgameSolver::Params solver;
  };

  explicit EndgameHastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game(const BeginGameRequest& req) override;

  // For the endgame benchmark, which reads the solve totals and toggles solver
  // features.
  EndgameTurnPolicy& endgame() { return endgame_; }

  // Build from `--player "--type=hastybot-endgame [options]"` tokens, with
  // --type and --name already stripped: every HastyBot option, plus the solver
  // Params under an "endgame-" prefix. Throws on bad input.
  static std::unique_ptr<EndgameHastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                         int thread_id, const std::string& name);

  static std::string options_help();

 private:
  EndgameTurnPolicy endgame_;
};

}  // namespace scribblez
