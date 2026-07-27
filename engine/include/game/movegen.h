#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/word_map.h"

#include <array>
#include <vector>

namespace scribblez {

// Selects the underlying enumeration algorithm. Both produce the same set of
// legal plays; GADDAG is the default and DAWG is retained for cross-validation.
enum class GenAlgo {
  GADDAG,  // Gordon's algorithm over the GADDAG (default; see docs/design.md).
  DAWG,    // Appel-Jacobson anchor + forward-trie traversal (reference impl).
};

// Generates all legal Scrabble moves for a (board, rack) pair, using
// cross-checks for perpendicular-word validity.
class MoveGenerator {
 public:
  MoveGenerator(const Board& board, const Dictionary& dict);

  // Every legal PLAY, scored. PASS and EXCHANGE are the agent layer's concern.
  std::vector<Move> generate(const Rack& rack, GenAlgo algo = GenAlgo::GADDAG);

  // Under the GADDAG algorithm, exactly the moves generate() emits while
  // scanning view-lane (`transposed`, `row`), in the same order and under the
  // same single-tile dedup rule -- so the union over all lanes equals
  // generate(), and an incremental caller can regenerate just the lanes a board
  // change touched.
  void generate_lane(const Rack& rack, bool transposed, int row, std::vector<Move>& out);

 private:
  const Board& board_;
  const Dictionary& dict_;
};

// MAGPIE-style move generation: instead of walking the GADDAG, look words up in
// `wm` by (length, letter multiset). Blank-free -- it generates no play that
// places a blank, so callers must restrict it to racks holding none; on those
// racks it produces exactly MoveGenerator::generate's plays. It exists to
// benchmark hash-anagram lookup against GADDAG traversal.
std::vector<Move> wmp_generate(const Board& board, const Dictionary& dict, const WordMap& wm,
                               const Rack& rack);

// Largest rack a play can place tiles from (one move plays 1..RACK_SIZE tiles).
inline constexpr int kMaxPlayTiles = 7;

// One GADDAG anchor plus, per tile count e, an admissible upper bound on the
// raw score of a play placing exactly e tiles here ("shadow play"). Bounding
// per tile count lets a caller pair each e-tile score with the best possible
// leave of size (rack - e), a tight equity bound for best-first pruning.
struct ShadowAnchor {
  bool transposed;
  int row;              // view-row of the lane
  int col;              // anchor column in the lane
  int last_anchor_col;  // nearest anchor to the left in this lane (-1 if none)
  std::array<int, kMaxPlayTiles + 1> score_bound_by_size;  // -1 where no play places e tiles
};

// The unit the MAGPIE-style shadow prices and the best-first loop generates: a
// group of words of one `length` laying down `placed` tiles, whose start column
// ranges over [leftmost_start_col, rightmost_start_col]. Every start in that
// range traverses the same playthrough multiset `pt`, so one WordMap scan of
// (pt + subrack) serves them all -- the found words slide across the range, each
// verified against the board at its own start. `score_bound` is an admissible
// upper bound (MAGPIE's descending-tile x descending-effective-multiplier bound)
// on the raw score of any play in the group.
struct ShadowExtent {
  bool transposed;
  int row;     // view-row of the lane
  int length;  // word length (the span covering each start is [s, s+length))
  int placed;  // tiles laid down (length minus the playthrough tile count)
  BitRack pt;  // playthrough multiset (from the rightmost start column)
  int score_bound;
  int8_t leftmost_start_col;
  int8_t rightmost_start_col;
};

// Best-first move generator: rather than enumerating every legal play, it
// bounds each anchor's per-tile-count score (shadow play) so a caller can rank
// anchors by an equity bound, generate anchor by anchor in descending order, and
// stop once no remaining anchor can beat the best move found. The plays are
// exactly MoveGenerator::generate's, split per anchor.
class ShadowMoveGen {
 public:
  ShadowMoveGen(const Board& board, const Dictionary& dict);

  // `rack` supplies the tile values the bounds assume.
  std::vector<ShadowAnchor> anchors(const Rack& rack) const;

  void generate_anchor(const ShadowAnchor& a, const Rack& rack, std::vector<Move>& out) const;

  // The finer-grained shadow the WordMap path drives: every word extent that
  // could hold a play, each with an admissible upper bound on its plays' raw
  // score, so low-value extents are never WordMap-scanned. Their plays union to
  // MoveGenerator::generate's.
  //
  // A non-null `wm` additionally prunes extents by word existence, which leaves
  // the move set unchanged. `nonplaythrough_has_word` is
  // `has_word[k] == true iff some size-k subrack forms a k-letter word`, for a
  // caller that has already computed it; pass nullptr to compute it here.
  std::vector<ShadowExtent> extents(
    const Rack& rack, const WordMap* wm = nullptr,
    const std::array<bool, kMaxPlayTiles + 1>* nonplaythrough_has_word = nullptr) const;

 private:
  const Board& board_;
  const Dictionary& dict_;
};

// Sub-multisets of a rack bucketed by tile count (blank-free): the candidate
// tile sets a WordMap play can place. Computed once per rack, reused across
// anchors.
using WmpSubracks = std::array<std::vector<BitRack>, kMaxPlayTiles + 1>;

// `rack_tiles` receives the rack's real-letter count; blanks are unrepresented.
void wmp_rack_subracks(const Rack& rack, WmpSubracks& out, int& rack_tiles);

// The GADDAG per-anchor generator's plays, via WordMap anagram lookups instead
// of traversal. Blank-free (see wmp_generate). The shadow best-first loop drives
// it, so only anchors that survive pruning are generated -- the regime where
// WordMap lookup beats GADDAG walking.
void wmp_generate_anchor(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         int rack_tiles, const ShadowAnchor& a, std::vector<Move>& out);

// The same for one extent: one WordMap scan of (playthrough + subrack) per
// subrack, each found word verified and placed at every start column of the
// group. Blank-free.
//
// `sub_terms`, when non-null, holds each size-`e.placed` subrack's equity
// contribution (leave value + pre-endgame term), so a subrack whose
// `e.score_bound + sub_terms[j]` cannot reach `best_equity` is skipped before
// its lookup.
void wmp_generate_extent(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         const ShadowExtent& e, std::vector<Move>& out, double best_equity = -1e18,
                         const double* sub_terms = nullptr);

}  // namespace scribblez
