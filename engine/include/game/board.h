#pragma once

#include "game/glyph.h"
#include "game/rack.h"
#include "game/tile_counts.h"

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

// Every letter A..Z allowed: the cross-check of a square with no perpendicular
// neighbor.
inline constexpr uint32_t kAllLettersMask = (1u << 26) - 1u;

// The perpendicular-word constraint on one empty square (Macondo's cross-set).
struct CrossCheck {
  uint32_t mask = kAllLettersMask;  // bit L set iff letter L is legal here
  int score = 0;                    // face value of the perpendicular run (blanks 0)
  bool has_neighbor = false;        // a perpendicular run touches this square
};

// Everything Board::apply(move, undo) changed, so Board::unapply(undo) can
// revert it exactly. Each write is logged as (location, old value) and undone
// in reverse order, which stays correct when one apply writes a location
// twice. The lists are vectors, not fixed arrays, because the first move on an
// empty board rewrites every cache entry.
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
  int num_tiles() const { return num_tiles_; }

  // The frame this board is expressed in: the game's natural frame (false) or
  // its diagonal transpose (true). Distinct from the `transposed` view argument
  // of the move-generation caches below, which picks a scanning orientation
  // within whichever frame the board is in.
  bool transposed() const { return transposed_; }

  // This board reflected across the main diagonal, (r,c) -> (c,r), with the
  // frame bit toggled; used for training-data symmetry augmentation. The
  // premium layout is diagonally symmetric, so the result is a legal position
  // that scores identically. The move-generation caches carry over by swapping
  // their two views, with no recomputation.
  Board transpose() const;

  Premium premium_at(int r, int c) const { return PREMIUM[r * BOARD_SIZE + c]; }

  // Places the move's new tiles; the move must be in this board's frame.
  // Valid move-generation caches are updated incrementally; stale ones are left
  // for a later ensure_movegen_caches().
  void apply(const Move& move);

  // As apply(move), also recording in `undo` what unapply() needs to restore
  // the board and its caches exactly. The endgame solver's make/unmake; racks
  // and draws are the caller's to undo.
  void apply(const Move& move, BoardUndo* undo);

  void unapply(const BoardUndo& undo);

  std::string to_string() const;

  // The tiles on this board, a designated blank counting as a blank.
  TileCounts tile_counts() const;

  // The tiles of the full distribution that are neither on this board nor in
  // `held`: what the holder of `held` cannot see. Throws if board + held
  // overdraw the distribution.
  TileCounts unseen_tiles(const Rack& held) const;

  // The number of tiles neither on this board nor among `held_tiles` rack
  // tiles: unseen_tiles(held).size() without the per-tile check.
  int unseen_count(int held_tiles) const;

  // The bag's size as the holder of `held_tiles` tiles sees it: unseen_count()
  // less the opponent's rack, which is full while the bag holds tiles.
  int pov_bag_size(int held_tiles) const;

  // unseen_tiles(known) as a rack: with an empty bag, exactly the other
  // player's rack. Throws as unseen_tiles does, or if that is more than a
  // rackful (the bag isn't empty).
  Rack hidden_rack(const Rack& known) const;

  // ---- Move-generation caches ----
  // Cross-checks and GADDAG anchors, as Macondo keeps them on its board: built
  // once, then updated incrementally by apply() so the generator never rescans
  // the whole board. Each is kept for two views, indexed in view coordinates:
  // the transposed view swaps row and column so every play is "horizontal".

  void ensure_movegen_caches(const Dictionary& dict) const;

  const std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE>& cross_checks(bool transposed) const {
    return cross_[transposed ? 1 : 0];
  }
  const std::array<bool, BOARD_SIZE * BOARD_SIZE>& gaddag_anchors(bool transposed) const {
    return ganchor_[transposed ? 1 : 0];
  }

  // One square's cross-check computed from scratch, in view coordinates; equal
  // to the cached entry. Requires a prior ensure_movegen_caches(), which binds
  // the dictionary.
  CrossCheck cross_check_at(bool transposed, int r, int c) const;

  static const std::array<Premium, BOARD_SIZE * BOARD_SIZE> PREMIUM;

 private:
  Glyph oriented_at(int r, int c, bool transposed) const;

  bool gaddag_anchor_at(bool transposed, int r, int c) const;

  // Inclusive [top, bot] row extent of the maximal filled perpendicular run
  // through the empty square (r, c).
  std::pair<int, int> perpendicular_run_bounds(bool transposed, int r, int c) const;

  // Letters that, placed at (r, c), form a word with the perpendicular run
  // through it. `prefix_node` is the DAWG node after the run above (r, c).
  uint32_t cross_check_letter_mask(bool transposed, int c, uint32_t prefix_node, int r,
                                   int bot) const;

  void recompute_all_caches() const;
  void update_caches_after_place(const std::pair<int, int>* placed, int n) const;

  // Every cache write goes through these, so `recorder_` captures exactly the
  // entries an update touches.
  void set_cross_(int transposed, int idx, const CrossCheck& cc) const;
  void set_anchor_(int transposed, int idx, bool value) const;

  std::array<Glyph, BOARD_SIZE * BOARD_SIZE> squares_{};
  int num_tiles_ = 0;
  bool transposed_ = false;

  // Mutable so the caches can be built lazily from const methods. `dict_` is
  // non-owning.
  mutable const Dictionary* dict_ = nullptr;
  mutable bool caches_valid_ = false;
  mutable std::array<CrossCheck, BOARD_SIZE * BOARD_SIZE> cross_[2];
  mutable std::array<bool, BOARD_SIZE * BOARD_SIZE> ganchor_[2]{};

  // Non-null only for the duration of one apply(move, &undo).
  mutable BoardUndo* recorder_ = nullptr;
};

}  // namespace scribblez

#include "inlines/game/board.inl"
