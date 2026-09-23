#include "belief/move_likelihood.h"

#include "agent/agent.h"
#include "lexicon/hasty_equity.h"

#include <algorithm>
#include <cmath>

namespace scribblez::belief {

namespace {

Rack kept_after(const Rack& rack, const Move& move) {
  Rack kept = rack;
  for (int i = 0; i < move.num_glyphs(); ++i) kept.remove(move.glyph(i).rack_tile());
  return kept;
}

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
  // A play is public down to the letter, and move generation is canonical, so
  // Move equality identifies it. An exchange discloses only its size.
  if (observed_.type() == MoveType::EXCHANGE) {
    return candidate.type() == MoveType::EXCHANGE &&
           candidate.num_glyphs() == observed_.num_glyphs();
  }
  return candidate == observed_;
}

void EquityLikelihood::explain(const Rack& rack, std::vector<ScoredLeave>* out) const {
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

  // Log-softmax with the maximum subtracted, so the exponentials stay in range.
  // The result stays in log space: at a low temperature an unlikely move can be
  // exp(-1000) from the best, and rounding that to zero would strike the
  // hypothesis out rather than rank it last.
  double total = 0.0;
  for (double e : equities) total += std::exp((e - best) / temperature_);
  const double log_total = std::log(total);

  for (size_t i = 0; i < moves.size(); ++i) {
    if (!matches_observation(moves[i])) continue;
    out->push_back({kept_after(rack, moves[i]), (equities[i] - best) / temperature_ - log_total});
  }
}

}  // namespace scribblez::belief
