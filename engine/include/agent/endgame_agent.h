#pragma once

#include "agent/agent.h"
#include "agent/endgame_turn_policy.h"
#include "endgame/endgame_solver.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// `Base` with an exact endgame: bag-empty turns go to an EndgameTurnPolicy,
// and every other turn, like any the solver declines, plays Base's move.
// Deriving from Base keeps the endgame variant usable wherever a Base is, e.g.
// as a rollout agent.
//
// Base supplies the command-line surface the endgame variant extends:
//   - Params, carrying the thread_id the pooled solver is keyed by;
//   - kType, its --type value (the variant's is kType + "-endgame");
//   - parse_params(tokens, thread_id, name, extra, type_label), parsing its
//     own options plus any registered in `extra` in one pass;
//   - options_help().
template <class Base>
class EndgameAgent : public Base {
 public:
  struct Params {
    typename Base::Params base;
    EndgameSolver::Params solver;
  };

  explicit EndgameAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void observe_move(const Move& move) override;
  void begin_game(const BeginGameRequest& req) override;

  // For the endgame benchmark, which reads the solve totals and toggles solver
  // features.
  EndgameTurnPolicy& endgame() { return endgame_; }

  // Build from `--player "--type=<kType>-endgame [options]"` tokens, with
  // --type and --name already stripped: every Base option, plus the solver
  // Params under an "endgame-" prefix. Throws on bad input.
  static std::unique_ptr<EndgameAgent> from_spec(const std::vector<std::string>& tokens,
                                                 int thread_id, const std::string& name);

  static std::string options_help();

 private:
  EndgameTurnPolicy endgame_;
};

}  // namespace scribblez

#include "inlines/agent/endgame_agent.inl"
