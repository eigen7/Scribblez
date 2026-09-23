#pragma once

// The likelihood half of Bayesian rack inference (docs/roadmap.md, track B):
// how plausible the opponent's observed action is if they held a given rack.
//
// The model: they chose by HastyBot static equity, softened by a temperature.
// P(action | rack) is a softmax over the equities of every move that rack
// could have made.
//
//   * The softmax is normalized per rack, over that rack's own moves, so this
//     is a true likelihood and hypotheses can be compared directly.
//   * On our own self-play data the model is nearly exact, since HastyBot plays
//     by equity. Against other opponents the temperature absorbs the mismatch,
//     which makes it the parameter worth sweeping.
//
// This class is the seam for other plausibility models. Macondo, for one,
// scores each hypothesis by mini-simulating it.

#include "belief/scored_leave.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <vector>

namespace scribblez {
class Dictionary;
}

namespace scribblez::belief {

class EquityLikelihood {
 public:
  // `board` and `bag_size` are as they stood before `observed` was made.
  // `temperature` is in equity points; as it approaches zero the model hardens
  // into "the observed move was the best one available to this rack".
  EquityLikelihood(const Board& board, const Dictionary& dict, int bag_size, const Move& observed,
                   double temperature);

  // The tiles the action made public, which every hypothesis rack must
  // contain: a play's placed tiles, or nothing for an exchange.
  const Rack& revealed() const { return revealed_; }

  // How many tiles a hypothesis supplies on top of revealed(); zero for a bingo.
  int hidden_tiles() const { return hidden_tiles_; }

  // Appends one entry per way the action could have arisen from `rack`: the
  // tiles it would have kept, weighted by log P(action | rack). A play has one
  // reading; an exchange has one per choice of tiles to surrender, since only
  // the count is public. Appends nothing when `rack` cannot explain the action.
  void explain(const Rack& rack, std::vector<ScoredLeave>* out) const;

 private:
  // Whether `candidate` would have looked to us exactly like what we saw.
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
