#pragma once

// The random bag-empty positions the endgame tests search over, shared by the
// solver suite, the EndgameHastyBot suite, and the NeuralAgent suite. A
// randomized corpus is what lets a test scan for the rare position that
// exercises it (a solver move that beats the greedy one, a proven loss, ...)
// instead of hand-building one.

#include "agent/agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/movegen.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"

#include <random>
#include <vector>

namespace scribblez {

// Few enough words that endgame branching stays low, but with enough hooks and
// overlaps that most random boards leave several plays available.
inline Dictionary tiny_dict() {
  return Dictionary::build_from_words({"CAT", "CATS", "AT",     "AS",     "BAT", "BATS", "HE",
                                       "TO",  "ON",   "NO",     "IT",     "IS",  "OAT",  "OATS",
                                       "HAT", "HATS", "RAT",    "RATS",   "DOG", "GOD",  "GO",
                                       "OD",  "DO",   "AERIES", "PARTIED"});
}

inline Rack random_rack(std::mt19937& rng) {
  Rack r;
  std::uniform_int_distribution<int> pick(0, 26);  // 26 -> blank
  for (int i = 0; i < RACK_SIZE; ++i) {
    const int v = pick(rng);
    r.add(v == 26 ? BLANK : Tile::of(v));
  }
  return r;
}

// A small endgame position: a board seeded with a few random plays, two short
// racks drawn from a curated tiny-dict letter set (so plays exist but branching
// stays low), and small random scores.
struct EndgamePos {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score;
  int opp_score;
};

inline EndgamePos random_endgame(std::mt19937& rng, const Dictionary& d, int rack_tiles) {
  static const char kLetters[] = "ATSOCHEBDGRINO";
  EndgamePos p;
  const int setup = std::uniform_int_distribution<int>(1, 3)(rng);
  for (int k = 0; k < setup; ++k) {
    const Rack seed = random_rack(rng);
    const std::vector<Move> plays = MoveGenerator(p.board, d).generate(seed);
    if (plays.empty()) break;
    p.board.apply(plays[std::uniform_int_distribution<size_t>(0, plays.size() - 1)(rng)]);
  }
  std::uniform_int_distribution<int> letter(0, static_cast<int>(sizeof(kLetters) - 2));
  for (int i = 0; i < rack_tiles; ++i) {
    p.my_rack.add(Tile::from_char(kLetters[letter(rng)]));
    p.opp_rack.add(Tile::from_char(kLetters[letter(rng)]));
  }
  p.my_score = std::uniform_int_distribution<int>(0, 40)(rng);
  p.opp_score = std::uniform_int_distribution<int>(0, 40)(rng);
  return p;
}

inline MoveRequest endgame_request(const EndgamePos& p, const Dictionary& d) {
  return MoveRequest{p.board, d, p.my_rack, p.opp_rack, p.my_score, p.opp_score, /*bag_size=*/0};
}

}  // namespace scribblez
