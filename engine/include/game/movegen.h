#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/word_map.h"

#include <array>
#include <vector>

namespace scribblez {

// Move generation, in three flavors that produce the same plays:
//   - MoveGenerator: exhaustive enumeration over the GADDAG (or the DAWG, as a
//     test reference).
//   - ShadowMoveGen + GADDAG: best-first by anchor, for finding the top-equity
//     play without generating the rest.
//   - ShadowMoveGen + WordMap (the wmp_* functions): best-first by word extent,
//     ported from MAGPIE. Faster, but blank-free, so racks with a blank use the
//     GADDAG variant.
// All read cross-checks and anchors from the Board's caches.

// Both produce the same plays; DAWG exists to cross-validate GADDAG in tests.
enum class GenAlgo {
  GADDAG,  // Gordon's algorithm
  DAWG,    // Appel-Jacobson: anchors plus forward-trie traversal
};

class MoveGenerator {
 public:
  MoveGenerator(const Board& board, const Dictionary& dict);

  // Every legal PLAY, scored. PASS and EXCHANGE are the agent layer's concern.
  std::vector<Move> generate(const Rack& rack, GenAlgo algo = GenAlgo::GADDAG);

  // Exactly the GADDAG moves generate() emits for one view lane, in the same
  // order, so a caller can regenerate only the lanes a board change touched.
  void generate_lane(const Rack& rack, bool transposed, int row, std::vector<Move>& out);

 private:
  const Board& board_;
  const Dictionary& dict_;
};

// Every play, found by WordMap lookups instead of a GADDAG walk. Blank-free: it
// never places a blank, so on a blank-free rack it matches
// MoveGenerator::generate. Tests use it to validate the WordMap path.
std::vector<Move> wmp_generate(const Board& board, const Dictionary& dict, const WordMap& wm,
                               const Rack& rack);

// Largest rack a play can place tiles from (one move plays 1..RACK_SIZE tiles).
inline constexpr int kMaxPlayTiles = 7;

// One GADDAG anchor with, per tile count e, an upper bound on the score of any
// play placing e tiles from it (a "shadow play"). Bounding per tile count lets
// a caller add the best leave of size rack - e, a tight equity bound.
struct ShadowAnchor {
  bool transposed;
  int row;              // view-row of the lane
  int col;              // anchor column in the lane
  int last_anchor_col;  // previous anchor in this lane (-1 if none); plays stop short of it
  std::array<int, kMaxPlayTiles + 1> score_bound_by_size;  // -1 where no play places e tiles
};

// The WordMap path's unit of work: the plays of one word `length` that place
// `placed` tiles and start anywhere in [leftmost_start_col,
// rightmost_start_col]. Every such start covers the same playthrough tiles
// `pt`, so one WordMap lookup of pt + subrack finds candidate words for all of
// them; each word is then checked against the board at each start.
// `score_bound` bounds the score of every play in the extent.
struct ShadowExtent {
  bool transposed;
  int row;  // view-row of the lane
  int length;
  int placed;  // length minus the playthrough tile count
  BitRack pt;
  int score_bound;
  int8_t leftmost_start_col;
  int8_t rightmost_start_col;
};

// Best-first move generation. It partitions the legal plays into units
// (anchors or extents), each with a score bound, so a caller can generate the
// units in descending bound order and stop once no remaining unit can beat the
// best play found. The units' plays together are exactly
// MoveGenerator::generate's.
class ShadowMoveGen {
 public:
  ShadowMoveGen(const Board& board, const Dictionary& dict);

  std::vector<ShadowAnchor> anchors(const Rack& rack) const;

  // The anchor's plays, via the GADDAG.
  void generate_anchor(const ShadowAnchor& a, const Rack& rack, std::vector<Move>& out) const;

  // A finer partition for the WordMap path (a port of MAGPIE's shadow walk):
  // every word extent that could hold a play. Generate one with
  // wmp_generate_extent().
  //
  // A non-null `wm` also drops extents that provably hold no word, without
  // changing the move set. `nonplaythrough_has_word[k]` says whether some
  // size-k subrack is a word; pass it if already computed, else nullptr.
  std::vector<ShadowExtent> extents(
    const Rack& rack, const WordMap* wm = nullptr,
    const std::array<bool, kMaxPlayTiles + 1>* nonplaythrough_has_word = nullptr) const;

 private:
  const Board& board_;
  const Dictionary& dict_;
};

// A rack's non-empty sub-multisets by size, blanks excluded: the tile sets a
// WordMap play can place. Compute once per turn.
using WmpSubracks = std::array<std::vector<BitRack>, kMaxPlayTiles + 1>;

// `rack_tiles` receives the rack's non-blank tile count.
void wmp_rack_subracks(const Rack& rack, WmpSubracks& out, int& rack_tiles);

// ShadowMoveGen::generate_anchor's plays via WordMap lookups. Blank-free.
void wmp_generate_anchor(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         int rack_tiles, const ShadowAnchor& a, std::vector<Move>& out);

// The plays of one extent. Blank-free.
//
// `sub_terms`, when non-null, gives each size-`e.placed` subrack's equity
// beyond score (e.g. leave value plus pre-endgame term), so subracks whose
// `e.score_bound + sub_terms[j]` cannot reach `best_equity` are skipped
// unlooked-up.
void wmp_generate_extent(const Board& board, const WordMap& wm, const WmpSubracks& subracks,
                         const ShadowExtent& e, std::vector<Move>& out, double best_equity = -1e18,
                         const double* sub_terms = nullptr);

}  // namespace scribblez
