#pragma once

#include "scribblez/board.h"
#include "scribblez/dictionary.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <array>
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

// Largest rack a play can place tiles from (one move plays 1..RACK_SIZE tiles).
inline constexpr int kMaxPlayTiles = 7;

// One GADDAG anchor plus, for each tile count e, an admissible upper bound on
// the raw score of a play that places exactly e tiles canonically anchored here
// ("shadow play"). score_bound_by_size[e] is -1 when no play places e tiles.
// Bounding per tile count lets a caller pair each e-tile score with the best
// possible leave of size (rack - e), giving a tight equity bound for best-first
// pruning.
struct ShadowAnchor {
  bool transposed;
  int row;              // view-row of the lane
  int col;              // anchor column in the lane
  int last_anchor_col;  // nearest anchor to the left in this lane (-1 if none)
  std::array<int, kMaxPlayTiles + 1> score_bound_by_size;
};

// Best-first move generator: instead of enumerating every legal play, it bounds
// each anchor's per-tile-count score (shadow play) so a caller can rank anchors
// by an equity bound, generate real moves anchor-by-anchor in descending order,
// and stop once no remaining anchor can beat the best move it has found.
// Produces exactly the same plays as MoveGenerator::generate (same build_play
// scoring + GADDAG enumeration), just split per anchor.
class ShadowMoveGen {
 public:
  ShadowMoveGen(const Board& board, const Dictionary& dict);

  // Every anchor with its per-tile-count score bounds. The board's move-gen
  // caches are ensured; `rack` supplies the tile values the bound assumes.
  std::vector<ShadowAnchor> anchors(const Rack& rack) const;

  // Append every legal PLAY canonically anchored at `a` to `out`.
  void generate_anchor(const ShadowAnchor& a, const Rack& rack, std::vector<Move>& out) const;

 private:
  const Board& board_;
  const Dictionary& dict_;
};

}  // namespace scribblez
