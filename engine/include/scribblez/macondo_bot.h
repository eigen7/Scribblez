#pragma once

#include "scribblez/agent.h"
#include "util/math.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

class WordMap;

// Total order on (equity, move) used to pick HastyBot's move: higher equity
// wins; exact-equity ties are broken by a canonical move ordering so the choice
// is deterministic and independent of move-generation order. Returns true iff
// (eq_a, a) is the better hasty choice.
bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b);

// Reference HastyBot selection: generate every legal play and take the
// hasty_move_better argmax. This is the specification the shadow-play search in
// HastyBotAgent::make_move reproduces, used to verify the optimized path.
Move hasty_best_move_reference(const MoveRequest& req);

// Same shadow-play search as HastyBotAgent::make_move's greedy path, but
// generates each anchor's plays with WordMap anagram lookups
// (wmp_generate_anchor) instead of GADDAG traversal. Picks the identical move.
// Blank-free: only valid on racks with no blanks (the WordMap path does not
// place blanks yet). Exists to benchmark shadow+WMP against shadow+GADDAG in
// HastyBot's real (pruned) regime.
Move hasty_best_move_wmp(const MoveRequest& req, const WordMap& wm);

// An in-process HastyBot player that ranks plays by static equity (score +
// leave value + opening/PEG/endgame adjustments) using the process-wide
// HastyEquity singleton. Thread-safe after HastyEquity::init() has been called.
//
// Move selection has two modes, set at construction:
//   temperature == 0 : pure argmax -- play the highest-equity move, found via a
//                      shadow-play search that bounds each anchor's best possible
//                      equity, processes anchors best-first, and stops once no
//                      remaining anchor can beat the move found (instead of
//                      generating every legal play). Ties broken by
//                      hasty_move_better.
//   temperature  > 0 : generate every legal play and sample
//                      softmax(equity / temperature) over the top-K moves by
//                      equity. This injects exploration into self-play data
//                      generation, which pure argmax play lacks (equity units
//                      are points, so a temperature of a few points spreads
//                      probability across near-best plays).
//
// Softmax sampling can be confined to the opening via `temperature_min_bag`:
// when set, the bot samples only while the bag holds at least that many tiles
// and plays argmax (the shadow-play search) once it falls below. This
// diversifies opening play (where near-equal plays differ in long-term value)
// while keeping HastyBot's exact strength through the rest of the game, where
// unconstrained sampling hurts.
class HastyBotAgent : public Agent {
 public:
  // Agent identity plus move-selection configuration.
  //   thread_id, name : the base Agent identity.
  //   temperature == 0 : pure argmax; `top_k` and `seed` are unused.
  //   temperature  > 0 : softmax-sample among the top `top_k` moves by equity,
  //                      with `seed` seeding the sampler.
  //   temperature_min_bag > 0 : sample only while `bag_size >= it`, playing
  //                             argmax below it; 0 samples for the whole game.
  // The defaults (off the identity fields) describe greedy HastyBot.
  struct Params {
    int thread_id = 0;
    std::string name;
    int top_k = 1;
    double temperature = 0.0;
    uint64_t seed = 0;
    int temperature_min_bag = 0;
  };

  explicit HastyBotAgent(const Params& params);

  Move make_move(const MoveRequest& req) override;
  bool supports_parallelism() const override { return true; }

  // Build a HastyBotAgent from `--player "--type=hastybot [options]"` tokens
  // (after the factory has stripped --type and --name). Optional --temperature=T
  // (0 = greedy argmax; > 0 = softmax sampling over the top-K), --top-k=K
  // (default 10), --temperature-min-bag=B (0 = sample all game; > 0 = sample
  // only while bag >= B), --seed=N (default: SeedProducer). Throws on bad input.
  static std::unique_ptr<HastyBotAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  int top_k_;
  double temperature_;
  int temperature_min_bag_;       // sample only while bag_size >= this (0 = all game)
  std::mt19937_64 rng_;           // drives softmax sampling when temperature_ > 0
  util::SoftmaxSampler sampler_;  // draws a move when temperature_ > 0
};

}  // namespace scribblez
