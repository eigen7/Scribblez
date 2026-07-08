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

// MAGPIE-style move generation: instead of walking the GADDAG, look words up in
// `wm` by (length, letter multiset). For a blank-free rack it produces the same
// set of legal plays as MoveGenerator::generate (same build_play scoring). Plays
// that place a blank are NOT generated, so callers must restrict it to racks with
// no blanks. It exists to benchmark hash-anagram lookup against GADDAG traversal.
std::vector<Move> wmp_generate(const Board& board, const Dictionary& dict, const WordMap& wm,
                               const Rack& rack);

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

// One MAGPIE shadow anchor, keyed by (playthrough_blocks, tiles_played): a word
// of `length` that lays down `placed` tiles over a fixed number of playthrough
// blocks, whose left edge (start column) ranges over [leftmost_start_col,
// rightmost_start_col]. Every start in that range traverses the same playthrough
// multiset `pt` (reconstructed from the rightmost start), so one WordMap scan of
// (pt + subrack) serves them all -- the found words slide across the start range,
// each verified against the board at its own start column. This is the unit the
// shadow prices and the best-first loop generates; `score_bound` is an admissible
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

  // Every word extent (a specific [wl, wr] span covering an anchor) that could
  // hold a play, each with an admissible upper bound on its plays' raw score.
  // This is the finer-grained shadow the WordMap path drives: a caller ranks ALL
  // extents by equity bound and generates them best-first, so low-value extents
  // are never WordMap-scanned. The union of every extent's plays equals
  // MoveGenerator::generate. When `wm` is non-null it is used for shadow-time
  // word-existence pruning (fewer extents); passing nullptr leaves the move set
  // unchanged, just less aggressively pruned. `nonplaythrough_has_word`, when
  // supplied, is `has_word[k] == true iff some size-k subrack forms a k-letter
  // word` -- a value the caller may already have computed; pass nullptr to have
  // extents() compute it from `wm`.
  std::vector<ShadowExtent> extents(
    const Rack& rack, const WordMap* wm = nullptr,
    const std::array<bool, kMaxPlayTiles + 1>* nonplaythrough_has_word = nullptr) const;

 private:
  const Board& board_;
  const Dictionary& dict_;
};

// Sub-multisets of a rack bucketed by tile count: index k holds every distinct
// size-k sub-multiset (blank-free), the candidate tile sets a WordMap play can
// place. Computed once per rack and reused across anchors.
using WmpSubracks = std::array<std::vector<BitRack>, kMaxPlayTiles + 1>;

// Enumerate `rack`'s sub-multisets into `out`; `rack_tiles` receives the rack's
// total real-letter count (blanks are not represented).
void wmp_rack_subracks(const Rack& rack, WmpSubracks& out, int& rack_tiles);

// Append every legal PLAY canonically anchored at `a` -- the same set as the
// GADDAG's generate_one_anchor -- using WordMap anagram lookups instead of GADDAG
// traversal. Blank-free: plays that place a blank are not generated, so callers
// must restrict it to racks with no blanks. This is the per-anchor WMP generator
// the shadow best-first loop drives; only the anchors that survive pruning are
// ever generated, which is the regime where WordMap lookup beats GADDAG walking.
void wmp_generate_anchor(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         int rack_tiles, const ShadowAnchor& a, std::vector<Move>& out);

// Append the legal PLAYs of one extent `e` using WordMap lookups: one scan of
// (playthrough + subrack) per subrack, each found word verified and placed at
// every start column in the group. A caller drives it best-first over extents so
// only the high-value ones are ever scanned. Blank-free.
//
// `sub_terms` (when non-null) holds, parallel to the rack's size-`e.placed`
// subracks, each subrack's equity contribution (leave value + pre-endgame term);
// a subrack whose `e.score_bound + sub_terms[j]` cannot reach `best_equity` is
// skipped before its lookup. Pass nullptr (the default) to scan every subrack.
void wmp_generate_extent(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         const ShadowExtent& e, std::vector<Move>& out, double best_equity = -1e18,
                         const double* sub_terms = nullptr);

}  // namespace scribblez
