#pragma once

#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <vector>

namespace scribblez {

// Selects the underlying enumeration algorithm. Both produce the same set of
// legal plays; GADDAG is the default and DAWG is retained for cross-validation.
enum class GenAlgo {
  GADDAG,  // Gordon's algorithm over the GADDAG (default; see docs/Scribblez.pdf).
  DAWG,    // Appel-Jacobson anchor + forward-trie traversal (reference impl).
};

// Generates all legal Scrabble moves for a (board, rack) pair, using cross-checks
// for perpendicular-word validity.
class MoveGenerator {
 public:
  MoveGenerator(const Board& board, const Dictionary& dict);

  // Returns all legal PLAY moves (with computed scores). Does NOT include
  // PASS or EXCHANGE actions; the agent layer is responsible for those.
  std::vector<Move> generate(const Rack& rack, GenAlgo algo = GenAlgo::GADDAG);

 private:
  const Board& board_;
  const Dictionary& dict_;
};

}  // namespace scribblez
