#pragma once

// A port of Macondo's Monte-Carlo simmer (montecarlo/montecarlo.go), the
// search behind its BestBot.
//
// Each iteration deals the opponent a random rack and plays every unpruned
// candidate out for a fixed number of plies of greedy static-equity moves
// (HastyBot's). A rollout then scores its final position two ways: the spread
// it gained plus the leaves the last two plies kept ("equity"), and a win
// probability, exact if the game ended, else read from Macondo's win-percentage
// table. The stopping rule (macondo_autostopper.h) prunes trailing candidates
// every 128 iterations and ends the sim; the best play by win probability wins.
//
// Every candidate of an iteration sees the same opponent rack and the same draw
// sequence (common random numbers), as in Macondo, where the candidates share
// one shuffled bag. Unlike Macondo, results are deterministic and independent
// of the thread count: iteration i is seeded by `seed + i`, and the stopping
// rule reads statistics reduced in iteration order.

#include "sim/macondo_autostopper.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <vector>

namespace scribblez {

class Dictionary;

class MacondoSimmer {
 public:
  struct Params {
    int plies = 2;  // rollout plies after the candidate itself
    int threads = 1;
    // Stop after this many iterations even if the stopping rule would go on;
    // 0 leaves it to the stopping rule alone, as in Macondo.
    int max_iterations = 0;
  };

  struct Result {
    std::vector<SimmedPlay> plays;  // best first (rank_simmed_plays)
    uint64_t iterations = 0;
  };

  // Sims `candidates` from `pos`, which must have tiles in the bag. Rollouts
  // stalemate at six consecutive scoreless turns, counting the
  // `scoreless_turns` already played before the decision. HastyEquity must be
  // initialized.
  static Result simulate(const Dictionary& dict, const SimPosition& pos,
                         const std::vector<Move>& candidates, int scoreless_turns,
                         const Params& params, uint64_t seed);
};

}  // namespace scribblez
