#pragma once

// What the pre-endgame solver asks of an endgame solver: the value of a
// bag-empty position. Behind this interface, any solver can evaluate the
// pre-endgame's leaves; SolverEndgameOracle adapts ours.
//
// Two oracles in the same pre-endgame solver need not agree: they classify some
// leaves differently, so the pre-endgame's choices differ with its oracle.

#include "endgame/endgame_solver.h"

#include <cstdint>

namespace scribblez {

// A bag-empty position's value for the side to move.
struct EndgameVerdict {
  static constexpr int kClassUnknown = EndgameResult::kClassUnknown;

  int cls = kClassUnknown;  // +1 win, 0 draw, -1 loss, or kClassUnknown
  int32_t spread = 0;       // final spread, exact or estimated; always set
  bool proven = false;      // `spread` is exact
};

enum class EndgameGoal {
  kClass,   // only the win/draw/loss class matters
  kSpread,  // the spread matters (the pre-endgame's tiebreak)
};

class EndgameOracle {
 public:
  virtual ~EndgameOracle() = default;

  // `effort` is a level the caller raises as it deepens: 1, 2, ... Each oracle
  // maps it to its own knob.
  virtual EndgameVerdict evaluate(const EndgameState& state, int effort, EndgameGoal goal) = 0;
};

// Our EndgameSolver as an oracle: effort is the solver's depth cap
// (Params::plies), under a fixed node budget per evaluation. A solve the budget
// declines outright gets its spread from a greedy playout (both sides playing
// HastyBot's move), so the verdict always carries an estimate; HastyEquity must
// be initialized.
class SolverEndgameOracle final : public EndgameOracle {
 public:
  // `solver` is borrowed and must outlive the oracle.
  SolverEndgameOracle(EndgameSolver& solver, uint64_t budget) : solver_(solver), budget_(budget) {}

  EndgameVerdict evaluate(const EndgameState& state, int effort, EndgameGoal goal) override;

 private:
  EndgameSolver& solver_;
  uint64_t budget_;
};

}  // namespace scribblez
