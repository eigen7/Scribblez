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

// Generate every legal play and take the hasty_move_better argmax. The
// specification hasty_best_move_wmp reproduces.
Move hasty_best_move_reference(const MoveRequest& req);

// The same move, without generating every legal play: bound each word extent's
// best possible equity (shadow play), generate extents best-first with WordMap
// anagram lookups, and stop once no remaining extent can beat the best move
// found. Racks holding a blank fall back to the GADDAG anchor search, the
// WordMap being blank-free.
Move hasty_best_move_wmp(const MoveRequest& req);

// An in-process HastyBot player that ranks plays by static equity (score +
// leave value + opening/PEG/endgame adjustments) using the process-wide
// HastyEquity singleton. Thread-safe once HastyEquity::init() has run.
//
// At temperature 0 it plays the argmax; above it, it samples
// softmax(equity / temperature) over the top-K plays, injecting the exploration
// that self-play data generation wants and pure argmax play lacks. Equity units
// are points, so a temperature of a few points spreads probability across
// near-best plays. `temperature_min_bag` confines sampling to the opening,
// where near-equal plays differ most in long-term value; below it the bot
// recovers HastyBot's exact strength.
class HastyBotAgent : public Agent {
 public:
  // The defaults describe greedy HastyBot.
  struct Params {
    int thread_id = 0;
    std::string name;
    int top_k = 1;
    double temperature = 0.0;
    uint64_t seed = 0;
    int temperature_min_bag = 0;  // sample only while bag_size >= this; 0 = all game
  };

  explicit HastyBotAgent(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  bool supports_parallelism() const override { return true; }

  // Build from `--player "--type=hastybot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  static std::string options_help();

  // Parse HastyBot's own options out of `tokens`, along with any a derived
  // agent registered in `extra`, in one pass. Lets a derived agent reuse the
  // option set instead of duplicating it. `type_label` names the type in
  // parse-error text.
  static Params parse_hasty_params(const std::vector<std::string>& tokens, int thread_id,
                                   const std::string& name,
                                   boost::program_options::options_description& extra,
                                   const char* type_label);

 private:
  int top_k_;
  double temperature_;
  int temperature_min_bag_;
  std::mt19937_64 rng_;
  util::SoftmaxSampler sampler_;
};

}  // namespace scribblez
