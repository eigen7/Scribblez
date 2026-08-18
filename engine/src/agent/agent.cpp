#include "agent/agent.h"

#include "game/movegen.h"

#include <utility>

namespace scribblez {

std::vector<Move> generate_legal_plays(const MoveRequest& req) {
  return MoveGenerator(req.board, req.dict).generate(req.my_rack);
}

namespace {

// One distinct tile type on a rack and how many copies of it are exchangeable.
struct ExchangeableType {
  Tile tile;
  int count;
};

// The exchange an odometer state describes: `chosen[k]` copies of type k.
Move exchange_move_of(const std::vector<ExchangeableType>& types, const std::vector<int>& chosen) {
  TileCounts pick;
  for (size_t k = 0; k < types.size(); ++k) {
    for (int c = 0; c < chosen[k]; ++c) pick.add(types[k].tile);
  }
  return Move::exchange(pick);
}

}  // namespace

std::vector<Move> generate_legal_exchanges(const MoveRequest& req) {
  std::vector<Move> out;
  if (req.bag_size < RACK_SIZE) return out;

  const TileCounts rack = req.my_rack.counts();
  std::vector<ExchangeableType> types;
  for (int i = 0; i <= BLANK; ++i) {
    const Tile t = Tile::of(i);
    if (rack.count(t) > 0) types.push_back({t, rack.count(t)});
  }

  // A mixed-radix odometer over how many copies of each type to exchange
  // (0..count per type): advancing before materializing skips the all-zero
  // combination, and a wrap past the last type means every combination has
  // been emitted.
  std::vector<int> chosen(types.size(), 0);
  while (true) {
    size_t j = 0;
    while (j < types.size() && ++chosen[j] > types[j].count) {
      chosen[j] = 0;
      ++j;
    }
    if (j == types.size()) break;
    out.push_back(exchange_move_of(types, chosen));
  }
  return out;
}

Move pick_uniform_random_play(const MoveRequest& req, std::mt19937_64& rng) {
  std::vector<Move> candidates = generate_legal_plays(req);
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  candidates.insert(candidates.end(), exchanges.begin(), exchanges.end());
  if (candidates.empty()) return Move::pass();
  std::uniform_int_distribution<size_t> d(0, candidates.size() - 1);
  return candidates[d(rng)];
}

Move exchange_or_pass(const MoveRequest& req) {
  return req.bag_size >= RACK_SIZE ? Move::exchange(req.my_rack.counts()) : Move::pass();
}

MoveDecision::MoveDecision(const Move& m, std::vector<Move> projected)
    : move(m), projected_remaining_moves(std::move(projected)) {}

}  // namespace scribblez
