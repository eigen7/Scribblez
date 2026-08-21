#pragma once

#include "game/glyph.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

constexpr int BOARD_SIZE = 15;
constexpr int CENTER = 7;

class Premium {
 public:
  enum Kind : uint8_t { kNone = 0, kDLS, kTLS, kDWS, kTWS };

  constexpr Premium() : kind_(kNone) {}
  constexpr explicit Premium(Kind k) : kind_(k) {}

  static const Premium NONE;
  static const Premium DLS;
  static const Premium TLS;
  static const Premium DWS;
  static const Premium TWS;

  constexpr bool operator==(Premium o) const { return kind_ == o.kind_; }
  constexpr bool operator!=(Premium o) const { return kind_ != o.kind_; }

  constexpr int letter_mult() const { return kind_ == kDLS ? 2 : kind_ == kTLS ? 3 : 1; }
  constexpr int word_mult() const { return kind_ == kDWS ? 2 : kind_ == kTWS ? 3 : 1; }

  constexpr char display_char() const;

  // "DL"/"TL"/"DW"/"TW", or nullptr for NONE.
  constexpr const char* code() const;

 private:
  Kind kind_;
};
static_assert(sizeof(Premium) == 1);

class Move;
class Dictionary;

// Bitmask with every letter A..Z allowed -- the cross-check of a square with no
// perpendicular neighbor.
inline constexpr uint32_t kAllLettersMask = (1u << 26) - 1u;

// The perpendicular-word constraint on a single empty square, mirroring
// Macondo's board-stored cross-sets.
struct CrossCheck {
  uint32_t mask = kAllLettersMask;  // bit L set iff letter L is legal here
  int score = 0;                    // sum of TILE_VALUES of the perpendicular run
  bool has_neighbor = false;        // true iff a perpendicular run touches this square
};

// Records everything Board::apply(move, undo) changes so Board::unapply(undo)
// can revert it exactly. Every mutation is logged as (location, old value) in
// write order and undone in reverse, which stays correct even when one apply
// touches a location twice. The lists are std::vectors because the first move
// on an empty board rewrites every cache entry, far more than an incremental
// update region.
struct BoardUndo {
  struct SquareRec {
    uint16_t idx;
    Glyph old;
  };
  struct CrossRec {
    uint8_t transposed;
    uint16_t idx;
    CrossCheck old;
  };
  struct AnchorRec {
    uint8_t transposed;
    uint16_t idx;
    bool old;
  };

  std::vector<SquareRec> squares;
  std::vector<CrossRec> crosses;
  std::vector<AnchorRec> anchors;
  bool prev_caches_valid = false;

  void clear() {
    squares.clear();
    crosses.clear();
    anchors.clear();
  }
};

class Board {
 public:
  Board();

  Glyph at(int r, int c) const { return squares_[r * BOARD_SIZE + c]; }
  void set(int r, int c, Glyph g);
  bool in_bounds(int r, int c) const;
  bool empty_board() const { return num_tiles_ == 0; }
  // Placed tiles currently on the board; every square write maintains it.
  int num_tiles() const { return num_tiles_; }

  Premium premium_at(int r, int c) const { return PREMIUM[r * BOARD_SIZE + c]; }

  // Place the move's new tiles. Valid move-generation caches are updated
  // incrementally; stale ones are left for a later ensure_movegen_caches().
  void apply(const Move& move);

  // As apply(move), but recording enough in `undo` for unapply() to restore the
  // board and its caches bit-for-bit. The make half of the endgame solver's
  // make/unmake; rack and draw bookkeeping stay the caller's.
  void apply(const Move& move, BoardUndo* undo);

  void unapply(const BoardUndo& undo);

  std::string to_string() const;

  // The tiles absent from both this board and `known` under the full tile
  // distribution -- with an empty bag, exactly the other player's rack (a
  // board blank counts as a blank tile). Throws when the remainder exceeds a
  // rackful (the bag is not empty) or when board + known overdraw the
  // distribution.
  Rack hidden_rack(const Rack& known) const;

  // ---- Persistent move-generation state ---------------------------------
  // Cross-checks and GADDAG anchors are computed once for the current board and
  // then maintained incrementally as moves are applied, so the generator need
  // not rescan the whole board each turn. Indexing is in the generator's view
  // coordinates: `transposed` swaps row and column.

  void ensure_movegen_caches(const Dictionary& dict) const;

  const std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>& cross_checks(bool transposed) const {
    return cross_[transposed ? 1 : 0];
  }
  const std::array<bool, BOARD_SIZE * BOARD_SIZE>& gaddag_anchors(bool transposed) const {
    return ganchor_[transposed ? 1 : 0];
  }

  // The cross-check of a single empty square, computed on demand from the live
  // board in view coordinates (`transposed` swaps row and column). Requires a
  // prior ensure_movegen_caches() to have bound the dictionary it reads; the
  // value equals cross_checks(transposed)[r * BOARD_SIZE + c], the cache entry
  // built from this same computation.
  CrossCheck cross_check_at(bool transposed, int r, int c) const;

  static const std::array<Premium, BOARD_SIZE * BOARD_SIZE> PREMIUM;

 private:
  Glyph oriented_at(int r, int c, bool transposed) const;

  // Per-square anchor computation (view coordinates).
  bool gaddag_anchor_at(bool transposed, int r, int c) const;

  // Inclusive [top, bot] row extent of the maximal filled perpendicular run
  // through the empty square (r, c).
  std::pair<int, int> perpendicular_run_bounds(bool transposed, int r, int c) const;

  // Letters that, placed at (r, c), complete a word with the perpendicular run
  // below it. `prefix_node` is the node reached by walking the run above.
  uint32_t cross_check_letter_mask(bool transposed, int c, uint32_t prefix_node, int r,
                                   int bot) const;

  void recompute_all_caches() const;
  void update_caches_after_place(const std::pair<int, int>* placed, int n) const;

  // Every cache write goes through these, so an active `recorder_` captures
  // exactly the entries an update touches.
  void set_cross_(int transposed, int idx, const CrossCheck& cc) const;
  void set_anchor_(int transposed, int idx, bool value) const;

  std::array<Glyph, BOARD_SIZE * BOARD_SIZE> squares_{};
  int num_tiles_ = 0;

  // Mutable so const accessors can lazily build the caches; `dict_` is
  // non-owning and outlives the board.
  mutable const Dictionary* dict_ = nullptr;
  mutable bool caches_valid_ = false;
  mutable std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE> cross_[2];
  mutable std::array<bool, BOARD_SIZE * BOARD_SIZE> ganchor_[2]{};

  // Non-null only for the duration of one apply(move, &undo).
  mutable BoardUndo* recorder_ = nullptr;
};

}  // namespace scribblez

#include "inlines/game/board.inl"
