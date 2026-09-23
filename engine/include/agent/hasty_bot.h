#pragma once

#include "agent/agent.h"
#include "util/math.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// Total order on (equity, move): higher equity wins, exact ties broken by a
// canonical move ordering so the choice never depends on generation order.
bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b);

// HastyBot's greedy move by brute force: generate every legal move and take
// the hasty_move_better argmax. The specification hasty_best_move_wmp is
// tested against.
Move hasty_best_move_reference(const MoveRequest& req);

// The same move as hasty_best_move_reference, found without generating every
// legal play (shadow-play bounds plus WordMap lookups). HastyBot's production
// greedy path.
Move hasty_best_move_wmp(const MoveRequest& req);

// A reimplementation of Macondo's HastyBot: ranks moves by static equity
// (score + leave value + opening, pre-endgame and endgame adjustments) from
// the process-wide HastyEquity tables.
//
// At temperature 0 it plays the argmax. Above 0 it samples
// softmax(equity / temperature) over the top-K moves, the exploration that
// self-play data generation wants. Equity is in points, so a temperature of a
// few points spreads probability across near-best moves.
class HastyBotAgent : public Agent {
 public:
  // The defaults describe greedy HastyBot.
  struct Params {
    int thread_id = 0;
    std::string name;
    int top_k = 1;
    double temperature = 0.0;
    uint64_t seed = 0;
  };

  explicit HastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;

  // Build from `--player "--type=hastybot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  static std::string options_help();

  // Parse HastyBot's options, plus any a derived agent registered in `extra`,
  // from `tokens` in one pass. `type_label` names the type in parse errors.
  static Params parse_hasty_params(const std::vector<std::string>& tokens, int thread_id,
                                   const std::string& name,
                                   boost::program_options::options_description& extra,
                                   const char* type_label);

 private:
  int top_k_;
  double temperature_;
  std::mt19937_64 rng_;
  util::SoftmaxSampler sampler_;
};

}  // namespace scribblez
