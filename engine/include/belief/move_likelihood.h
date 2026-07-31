#pragma once

// The likelihood half of Bayesian rack inference (docs/roadmap.md, track B):
// how plausible the opponent's observed action is if they held a given rack.
//
// The model is that they chose by HastyBot static equity, softened by a
// temperature -- P(action | rack) is a softmax over the equities of every move
// that rack could have made. Two properties of that shape matter to callers:
//
//   * The softmax denominator is per-rack, since different hypotheses can make
//     different moves. This is therefore a genuine normalized likelihood, not
//     a score defined up to a constant shared across hypotheses, and comparing
//     two hypotheses is only meaningful because each was normalized over its
//     own move set.
//   * On our own self-play data the model is not an approximation. HastyBot
//     *is* the equity argmax, so temperature is the only thing standing
//     between this and the exact process that generated the move. Against any
//     other opponent the temperature absorbs the mismatch, which is what makes
//     it the parameter worth sweeping.
//
// The plausibility model is the part meant to be swapped: Macondo scores each
// hypothesis by mini-simulating it instead, and pricing that against this is
// the reason to keep the seam visible.

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <vector>

namespace scribblez {
class Dictionary;
}

namespace scribblez::belief {

// One way an observed action could have arisen from a hypothesis rack: the
// tiles it would have left the opponent holding, and P(action | rack) for that
// explanation. A tile play has exactly one explanation, its own leave. An
// exchange has one per distinct choice of tiles to surrender, because only the
// number exchanged is public.
struct Explanation {
  Rack kept;
  double probability;
};

class EquityLikelihood {
 public:
  // `board` is the board as it stood before `observed` was made and
  // `bag_size` the bag at that moment. `temperature` is in equity points; as
  // it approaches zero the model hardens into "the observed move was the best
  // one available to this rack".
  EquityLikelihood(const Board& board, const Dictionary& dict, int bag_size, const Move& observed,
                   double temperature);

  // The tiles the action put on public display, which every hypothesis rack
  // must therefore contain: a play's placed tiles, or nothing at all for an
  // exchange, whose tiles stay hidden.
  const Rack& revealed() const { return revealed_; }

  // How many tiles a hypothesis has to supply on top of revealed(). Zero means
  // the action left nothing to infer -- a bingo empties the rack.
  int hidden_tiles() const { return hidden_tiles_; }

  // Appends every explanation of the observed action under `rack`, leaving
  // `out` untouched when the rack cannot explain the action at all.
  void explain(const Rack& rack, std::vector<Explanation>* out) const;

 private:
  // Whether `candidate` would have looked, to us, exactly like what we saw.
  bool matches_observation(const Move& candidate) const;

  const Board& board_;
  const Dictionary& dict_;
  int bag_size_;
  Move observed_;
  double temperature_;
  Rack revealed_;
  int hidden_tiles_;
};

}  // namespace scribblez::belief
