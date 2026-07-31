#include "belief/move_likelihood.h"

#include "agent/agent.h"
#include "lexicon/hasty_equity.h"

#include <algorithm>
#include <cmath>

namespace scribblez::belief {

namespace {

// The rack left over once `move` has taken its tiles from `rack`.
Rack kept_after(const Rack& rack, const Move& move) {
  Rack kept = rack;
  for (int i = 0; i < move.num_glyphs(); ++i) kept.remove(move.glyph(i).rack_tile());
  return kept;
}

// The tiles `move` took off the rack.
Rack tiles_used(const Move& move) {
  Rack used;
  for (int i = 0; i < move.num_glyphs(); ++i) used.add(move.glyph(i).rack_tile());
  return used;
}

}  // namespace

EquityLikelihood::EquityLikelihood(const Board& board, const Dictionary& dict, int bag_size,
                                   const Move& observed, double temperature)
    : board_(board),
      dict_(dict),
      bag_size_(bag_size),
      observed_(observed),
      temperature_(temperature),
      revealed_(observed.type() == MoveType::PLAY ? tiles_used(observed) : Rack{}),
      hidden_tiles_(RACK_SIZE - revealed_.size()) {}

bool EquityLikelihood::matches_observation(const Move& candidate) const {
  // A play is public down to the letter, so it identifies itself: every byte
  // of a Move is meaningful and the generators produce canonical orderings.
  // An exchange discloses only how many tiles went back, so every exchange of
  // that size is an equally valid reading of what we saw.
  if (observed_.type() == MoveType::EXCHANGE) {
    return candidate.type() == MoveType::EXCHANGE &&
           candidate.num_glyphs() == observed_.num_glyphs();
  }
  return candidate == observed_;
}

void EquityLikelihood::explain(const Rack& rack, std::vector<Explanation>* out) const {
  const Rack opp_rack;  // their opponent is us, and our rack was hidden from them
  // Neither move generation nor static equity reads the scores.
  const MoveRequest req{board_, dict_, rack, opp_rack, 0, 0, bag_size_};
  std::vector<Move> moves = generate_legal_plays(req);
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  moves.insert(moves.end(), exchanges.begin(), exchanges.end());
  if (moves.empty()) return;

  const std::vector<double> equities =
    HastyEquity::instance().equities(moves, board_, bag_size_, opp_rack, rack);
  const double best = *std::max_element(equities.begin(), equities.end());

  // Softmax with the maximum subtracted out, so the exponentials stay in
  // range however far apart the equities spread.
  double total = 0.0;
  for (double e : equities) total += std::exp((e - best) / temperature_);

  for (size_t i = 0; i < moves.size(); ++i) {
    if (!matches_observation(moves[i])) continue;
    const double p = std::exp((equities[i] - best) / temperature_) / total;
    if (p > 0.0) out->push_back({kept_after(rack, moves[i]), p});
  }
}

}  // namespace scribblez::belief
