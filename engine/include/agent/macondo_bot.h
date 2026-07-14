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

// Total order on (equity, move) used to pick HastyBot's move: higher equity
// wins; exact-equity ties are broken by a canonical move ordering so the choice
// is deterministic and independent of move-generation order. Returns true iff
// (eq_a, a) is the better hasty choice.
bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b);

// Reference HastyBot selection: generate every legal play and take the
// hasty_move_better argmax. This is the specification the shadow-play search in
// HastyBotAgent::make_move reproduces, used to verify the optimized path.
Move hasty_best_move_reference(const MoveRequest& req);

// HastyBotAgent::make_move's greedy path: bound each word extent's best
// possible equity (shadow play), generate extents best-first with WordMap
// anagram lookups (wmp_generate_extent), and stop once no remaining extent can
// beat the best move found. Racks holding a blank fall back to the GADDAG
// anchor search (the WordMap is blank-free). Picks exactly the
// hasty_best_move_reference move.
Move hasty_best_move_wmp(const MoveRequest& req);

// An in-process HastyBot player that ranks plays by static equity (score +
// leave value + opening/PEG/endgame adjustments) using the process-wide
// HastyEquity singleton. Thread-safe after HastyEquity::init() has been called.
//
// Move selection has two modes, set at construction:
//   temperature == 0 : pure argmax -- play the highest-equity move, found via
//                      hasty_best_move_wmp: a shadow-play search that bounds
//                      each word extent's best possible equity, generates
//                      extents best-first with WordMap anagram lookups, and
//                      stops once no remaining extent can beat the move found
//                      (instead of generating every legal play). Ties broken
//                      by hasty_move_better.
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

  MoveDecision make_move(const MoveRequest& req) override;
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

  // Parse the shared HastyBot options (--top-k, --temperature,
  // --temperature-min-bag, --seed) out of `tokens`, together with any extra
  // options the caller registered in `extra` (parsed in the same pass), into a
  // Params. `thread_id`/`name` set the returned identity; the seed defaults to
  // SeedProducer when --seed is absent, and the equity tables are lazily loaded
  // for the active lexicon. `type_label` names the type in parse-error text.
  // Derived agents call this to reuse HastyBot's option set without duplicating
  // it, extending the parse with their own flags via `extra`.
  static Params parse_hasty_params(const std::vector<std::string>& tokens, int thread_id,
                                   const std::string& name,
                                   boost::program_options::options_description& extra,
                                   const char* type_label);

 private:
  int top_k_;
  double temperature_;
  int temperature_min_bag_;       // sample only while bag_size >= this (0 = all game)
  std::mt19937_64 rng_;           // drives softmax sampling when temperature_ > 0
  util::SoftmaxSampler sampler_;  // draws a move when temperature_ > 0
};

}  // namespace scribblez
