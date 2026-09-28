#pragma once

// A port of Macondo's pre-endgame solver (preendgame/peg.go, peg_generic.go):
// with a few tiles in the bag, it ranks the mover's plays by how many of the
// possible bag contents they win, counting a draw as half.
//
// For each play and each ordered draw the bag might hold, the solver plays the
// game out: through every opponent reply while the bag still holds tiles
// (pessimistically: a draw is lost if any reply beats it), through a nested
// solve when it is the mover's turn again with tiles still in the bag, and
// through the endgame oracle once the bag is empty. Ties at the top go to the
// plays that empty the bag, ranked by total spread over fully solved endgames.
// The whole search repeats at rising oracle effort (Macondo's iterative
// deepening over endgame plies), each pass ordered by the last one's ranking.
//
// Single-threaded, where Macondo splits the plays across threads; the only
// difference is speed.

#include "endgame/endgame_oracle.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

class Dictionary;

// The mover's decision point. The bag holds 1 to PreEndgameSolver::kMaxInBag
// tiles.
struct PreEndgamePosition {
  const Dictionary* dict = nullptr;
  Board board;
  Rack my_rack;  // full
  // Tiles known to be on the opponent's rack (face-up leaves); they are never
  // in the bag.
  Rack opp_known;
  int my_score = 0;
  int opp_score = 0;
  int scoreless_turns = 0;  // consecutive scoreless turns already played
};

// How a leaf the oracle could not classify is scored.
enum class UnprovenPolicy {
  kEstimatedSpread,  // by the sign of the oracle's spread estimate
  kLoss,             // as a loss for the mover, whatever the estimate
};

class PreEndgameSolver {
 public:
  static constexpr int kMaxInBag = 6;
  // Macondo's cap on the tied plays the spread tiebreak solves.
  static constexpr int kTiebreakPlays = 20;

  struct Params {
    // The highest oracle effort; 0 picks Macondo's schedule from the spread
    // (macondo_max_effort).
    int max_effort = 0;
    UnprovenPolicy unproven = UnprovenPolicy::kEstimatedSpread;

    // Register the options under `prefix`, bound to this object's fields,
    // whose current values become the defaults.
    void add_options(boost::program_options::options_description& desc, const std::string& prefix);
  };

  // One play's standing. `points` counts the ordered bag draws it wins, a draw
  // counting half; `total` is the number of draws it was scored over.
  struct RankedPlay {
    Move move;
    double points = 0.0;
    int total = 0;
  };

  // Macondo's BestBot sets its endgame plies from the mover's spread: the
  // further behind or ahead, the shallower, since a bingo out decides it.
  static int macondo_max_effort(int spread);

  // `oracle` is borrowed and must outlive the solver.
  explicit PreEndgameSolver(EndgameOracle& oracle) : oracle_(oracle) {}

  // The mover's plays (and PASS; no exchanges), best first. HastyEquity must be
  // initialized.
  std::vector<RankedPlay> solve(const PreEndgamePosition& pos, const Params& params);

 private:
  EndgameOracle& oracle_;
};

}  // namespace scribblez
