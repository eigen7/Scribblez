#pragma once

#include "game/glyph.h"

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

  // Returns "DL"/"TL"/"DW"/"TW" for premium squares, nullptr for NONE.
  constexpr const char* code() const;

 private:
  Kind kind_;
};
static_assert(sizeof(Premium) == 1);

class Move;        // forward declaration
class Dictionary;  // forward declaration

// Bitmask with every letter A..Z allowed (the default cross-check).
inline constexpr uint32_t kAllLettersMask = (1u << 26) - 1u;

// Move-generation cross-check for a single empty square: which letters may be
// placed there (perpendicular-word validity), the score contributed by the
// existing perpendicular run, and whether such a run exists at all. This is the
// persistent, board-resident state that the move generator reads each turn
// (mirroring Macondo's board-stored cross-sets).
struct CrossCheck {
  uint32_t mask = kAllLettersMask;  // bit L set iff letter L is legal here
  int score = 0;                    // sum of TILE_VALUES of the perpendicular run
  bool has_neighbor = false;        // true iff a perpendicular run touches this square
};

// Records everything Board::apply(move, undo) changes so Board::unapply(undo)
// can revert it exactly: each overwritten square, each overwritten cache entry
// (cross-check or GADDAG anchor, per orientation), and the prior validity flag.
// Every mutation is logged as (location, old value) in write order, so undo
// restores them in reverse -- correct even when one apply touches a location
// more than once. The first move on an empty board rewrites every cache entry
// (a full recompute), so the entry lists are std::vectors rather than a
// fixed-capacity buffer sized for one incremental update region.
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
  bool empty_board() const;

  Premium premium_at(int r, int c) const { return PREMIUM[r * BOARD_SIZE + c]; }

  // Place the move's new tiles on the board. If the move-generation caches are
  // valid, they are updated incrementally for the placed tiles; otherwise they
  // are left invalid for a later full rebuild via ensure_movegen_caches().
  void apply(const Move& move);

  // As apply(move), but records every square, cache entry, and validity-flag
  // change into `undo` so a subsequent unapply(undo) restores the exact prior
  // state (squares, both cross-check tables, both anchor tables, and the
  // caches-valid flag). This is the make half of the endgame solver's
  // make/unmake; there is no draw/rack bookkeeping here -- that is the caller's.
  void apply(const Move& move, BoardUndo* undo);

  // Revert the changes recorded in `undo` by a prior apply(move, &undo),
  // restoring the board and its move-generation caches bit-for-bit.
  void unapply(const BoardUndo& undo);

  // Pretty-print the board to a string.
  std::string to_string() const;

  // ---- Persistent move-generation state ---------------------------------
  // Cross-checks and GADDAG anchors are computed once for the current board
  // and then maintained incrementally as moves are applied, so the generator
  // need not rescan the whole board each turn. Indexing is in the generator's
  // view coordinates: `transposed == false` for horizontal plays (board
  // coordinates) and `transposed == true` for vertical plays (row/col swapped).

  // Build (or rebuild) the caches for the current board if they are stale.
  void ensure_movegen_caches(const Dictionary& dict) const;

  const std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>& cross_checks(bool transposed) const {
    return cross_[transposed ? 1 : 0];
  }
  const std::array<bool, BOARD_SIZE * BOARD_SIZE>& gaddag_anchors(bool transposed) const {
    return ganchor_[transposed ? 1 : 0];
  }

  static const std::array<Premium, BOARD_SIZE * BOARD_SIZE> PREMIUM;

 private:
  // Board square read in the given orientation (transposed swaps row/col).
  Glyph oriented_at(int r, int c, bool transposed) const;

  // Per-square cache computations (view coordinates; `dict_` must be set).
  CrossCheck cross_check_at(bool transposed, int r, int c) const;
  bool gaddag_anchor_at(bool transposed, int r, int c) const;

  // Inclusive [top, bot] row extent of the maximal filled perpendicular run
  // through the empty square (r, c) at fixed column c.
  std::pair<int, int> perpendicular_run_bounds(bool transposed, int r, int c) const;

  // Bitmask (bit L set) of letters that, placed at (r, c), form a word accepted
  // by the dictionary together with the perpendicular run below it. prefix_node
  // is the node reached by walking the run above (r, c); bot is its bottom row.
  uint32_t cross_check_letter_mask(bool transposed, int c, uint32_t prefix_node, int r,
                                   int bot) const;

  // Full and incremental cache maintenance.
  void recompute_all_caches() const;
  void update_caches_after_place(const std::pair<int, int>* placed, int n) const;

  // Cache-entry writers that log the prior value to `recorder_` when one is
  // active, so make/unmake captures exactly the entries an update touches.
  void set_cross_(int transposed, int idx, const CrossCheck& cc) const;
  void set_anchor_(int transposed, int idx, bool value) const;

  std::array<Glyph, BOARD_SIZE * BOARD_SIZE> squares_{};

  // Cache state. Mutable so const accessors (used on const Board&) can lazily
  // build them; `dict_` is non-owning and outlives the board.
  mutable const Dictionary* dict_ = nullptr;
  mutable bool caches_valid_ = false;
  mutable std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE> cross_[2];
  mutable std::array<bool, BOARD_SIZE * BOARD_SIZE> ganchor_[2]{};

  // When non-null, the setters below log each overwritten cache entry here so
  // apply(move, &undo) can be reverted. Only set for the duration of one apply.
  mutable BoardUndo* recorder_ = nullptr;
};

}  // namespace scribblez

#include "inlines/game/board.inl"
