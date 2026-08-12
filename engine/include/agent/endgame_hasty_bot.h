#pragma once

#include "agent/endgame_turn_policy.h"
#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// A HastyBot that hands the endgame off to an EndgameTurnPolicy, so it
// converts won endgames and defends lost ones optimally rather than greedily.
// Until the bag empties it plays exactly like HastyBotAgent, as it does on any
// turn the solver declines.
class EndgameHastyBotAgent : public HastyBotAgent {
 public:
  struct Params {
    HastyBotAgent::Params hasty;
    EndgameSolver::Params solver;
  };

  explicit EndgameHastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game(std::array<int, 2> initial_scores) override;

  // Exposed for the endgame benchmark, which reads the solve totals and toggles
  // individual solver features.
  EndgameTurnPolicy& endgame() { return endgame_; }

  // Build from `--player "--type=hastybot-endgame [options]"` tokens, with
  // --type and --name already stripped. Accepts every HastyBot option plus the
  // solver Params under an "endgame-" prefix. Throws on bad input.
  static std::unique_ptr<EndgameHastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                         int thread_id, const std::string& name);

  static std::string options_help();

 private:
  EndgameTurnPolicy endgame_;
};

}  // namespace scribblez
