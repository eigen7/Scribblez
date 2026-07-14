#pragma once

#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace scribblez {

class Dictionary;

// Result of a single endgame solve, from the perspective of the side to move
// at the solved position ("the solving side").
struct EndgameResult {
  Move best;                // best move for the solving side
  int32_t value = 0;        // optimal final spread (solving score - opponent
                            // score, after end-of-game adjustments)
  int depth_completed = 0;  // deepest fully-completed iterative-deepening depth
                            // (0: the solve was declined up front or the budget
                            // ran out inside the first iteration, so `best` is
                            // only the statically best-estimated root move)
  uint64_t nodes = 0;       // negamax entries + greedy-playout plies spent
  bool proven = false;      // true iff `value` rests entirely on real game ends,
                            // with no greedy-playout leaf in the lines that
                            // determined it: then `value` is the exact
                            // game-theoretic spread (in first_win mode, a true
                            // win/draw/loss verdict), not a depth-limited
                            // estimate. Declined solves (depth_completed == 0)
                            // are never proven.
};

// Exact/near-exact endgame solver for a pre-endgame position: bag empty and both
// racks known. It is a self-contained negamax search (alpha-beta + principal
// variation search + iterative deepening + a transposition table) with a greedy
// leaf playout, and depends only on (board, dictionary, racks, scores) -- no
// equity model or leave tables.
//
// Scoreless-turn handling is compressed relative to the full game rule (six
// consecutive zero-score turns end the game as a stalemate). Because a pass
// leaves the position unchanged, a chain of passes beyond the first adds no new
// information to the search, so the solver ends the game after an internal cap
// of two consecutive scoreless turns (each side then subtracts its own rack
// value). At solve entry the caller's real scoreless-turn count is rebased to 1
// when positive and 0 otherwise. Any play resets the internal count to 0,
// mirroring the real game (even a zero-scoring play).
//
// A search result is "proven" when its value was derived entirely from real
// game ends -- no greedy-playout leaf entered the lines that determined it. A
// proven full-window value is the exact game-theoretic spread, so iterative
// deepening stops the moment an iteration returns one; a proven first_win value
// is a true win/draw/loss verdict, so deepening stops once the class is settled.
// Proven-ness rides along every search result and every transposition-table
// entry, propagated through the negamax recursion (a node is proven only if the
// child searches its verdict rests on are all proven).
//
// TODO(multithreading): when single-game (non-parallel-self-play) settings
// arrive, add an opt-in threaded mode: repack TTEntry into two XOR-verified
// atomic uint64 words (the lockless Hyatt scheme MAGPIE uses, which also
// shrinks the entry), then run shared-TT lazy-SMP -- helper threads search at
// staggered depths with jittered root orderings and only populate the TT,
// while the main thread owns the PV. Keep 1 thread the default: both Macondo
// ("~2x with 3 threads, degrades beyond") and MAGPIE found endgame SMP's
// returns modest, so stop at lazy-SMP unless a profile says otherwise.
class EndgameSolver {
 public:
  // tt_log2_entries sizes the transposition table to 2^tt_log2_entries entries.
  explicit EndgameSolver(int tt_log2_entries = 16);

  // Solve the position for the side holding my_rack (to move). Both racks must
  // be the actual remaining tiles (bag empty). scoreless_turns is the number of
  // consecutive zero-score turns already played in the real game.
  //
  // With first_win, the root alpha-beta window is pinned to
  // (kFirstWinAlpha, kFirstWinBeta) around an even final spread, so the search
  // only resolves the win/draw/loss class instead of the exact spread -- proofs
  // arrive far cheaper, at the price of an arbitrary (not spread-maximal)
  // winning move. A result value >= kFirstWinBeta proves a win to the searched
  // depth, <= kFirstWinAlpha proves every move loses (the best move is then
  // meaningless), and 0 holds a draw.
  //
  // The search looks at most max_plies deep, and node_budget is a hard cap on nodes spent
  // (exceeded by at most one greedy playout's plies before the abort lands).
  // A position with more root moves than node_budget provably cannot complete
  // its first iteration, so the solve is declined immediately after root move
  // generation rather than burning the budget on a fraction of the root. A
  // legal move is always returned: the last completed iteration's best, else the
  // best fully-searched root move of the partial first iteration, else the
  // estimate-ordered top root move.
  EndgameResult solve(const Board& board, const Dictionary& dict, const Rack& my_rack,
                      const Rack& opp_rack, int my_score, int opp_score, int scoreless_turns,
                      uint64_t node_budget, int max_plies, bool first_win = false);

  // The first-win root window: a final spread >= kFirstWinBeta is a win,
  // <= kFirstWinAlpha a loss, 0 a draw.
  static constexpr int32_t kFirstWinAlpha = -1;
  static constexpr int32_t kFirstWinBeta = 1;

  // Invalidate every transposition-table entry (call between games; entries are
  // spread-rebased and thus reusable across turns within one game, but not
  // across games). O(1): it bumps the table's generation counter, and probes
  // treat entries from older generations as empty.
  void clear();

 private:
  // Bound type in the low two bits of a TTEntry's flag byte; kEmpty == 0 marks a
  // never-written slot. Every stored entry carries a real bound (>= kExact), so a
  // nonzero flag byte always means "occupied".
  enum TTFlag : uint8_t { kEmpty = 0, kExact, kLower, kUpper };
  static constexpr uint8_t kBoundMask = 0x03;  // bits 0-1: the TTFlag bound type
  static constexpr uint8_t kProvenBit = 0x04;  // bit 2: the stored value is proven
  static constexpr uint8_t tt_bound(uint8_t flag) { return flag & kBoundMask; }
  static constexpr bool tt_is_proven(uint8_t flag) { return (flag & kProvenBit) != 0; }
  static constexpr uint8_t tt_pack(uint8_t bound, bool proven) {
    return static_cast<uint8_t>(bound | (proven ? kProvenBit : 0));
  }

  // A transposition-table entry (32 bytes): the full hash (to reject index
  // collisions), the best move for ordering, the search value stored relative
  // to the node's spread (so it is reusable at any absolute score), the flag
  // byte (bound type in bits 0-1, proven bit in bit 2), the search depth that
  // produced it, and the table generation that wrote it (entries from older
  // generations read as empty; see clear()).
  struct TTEntry {
    uint64_t hash = 0;
    Move best;
    int32_t score_rel = 0;
    uint8_t flag = kEmpty;
    uint8_t depth = 0;
    uint16_t gen = 0;
  };

  // A negamax return: the search value together with whether it is proven (rests
  // entirely on real game ends, so it is exact rather than a playout estimate).
  struct SearchResult {
    int32_t value = 0;
    bool proven = false;
  };

  // Everything one make() changes, so unmake() can restore it: the board undo,
  // both scores, the scoreless count, the game-over flag, the incremental board
  // hash, the side that moved, and the move (to return its tiles to the rack).
  struct Frame {
    BoardUndo board_undo;
    int32_t scores[2] = {0, 0};
    int scoreless = 0;
    bool game_over = false;
    uint64_t board_hash = 0;
    int mover = 0;
    Move move;
  };

  // --- Iterative deepening / search ---------------------------------------
  SearchResult run_root(int depth, int32_t alpha, int32_t beta,
                        std::vector<std::pair<Move, int32_t>>& root_moves, Move* best_out);
  SearchResult negamax(int depth, int32_t alpha, int32_t beta, int ply);
  // Search one child at (alpha, beta) and return its value from the parent's
  // perspective (negated) with the child's proven bit, using a full window for
  // the first child and a scout-plus-conditional-re-search for the rest (PVS).
  SearchResult search_child(int child_depth, int32_t alpha, int32_t beta, int child_ply,
                            bool first);
  int32_t greedy_playout(uint64_t node_key, int ply);

  // --- Make / unmake ------------------------------------------------------
  void make(const Move& move, int ply);
  void unmake(int ply);
  void xor_play_hash(const Move& move);

  // --- Position keys / evaluation helpers ---------------------------------
  uint64_t compute_board_hash() const;
  uint64_t node_hash() const;
  int32_t spread_stm() const { return scores_[stm_] - scores_[1 - stm_]; }

  // --- Move ordering / greedy choice --------------------------------------
  int32_t order_estimate(const Move& move, const Move& tt_move, bool have_tt_move) const;
  void order_moves(std::vector<std::pair<Move, int32_t>>& moves, const Move& tt_move,
                   bool have_tt_move) const;
  const Move& greedy_pick(const std::vector<Move>& plays) const;
  double playout_adjusted(const Move& move) const;
  int placed_face_value(const Move& move) const;

  // --- Transposition table ------------------------------------------------
  TTEntry* tt_probe(uint64_t hash);
  void tt_store(uint64_t hash, int32_t score_rel, uint8_t bound, bool proven, const Move& best,
                int depth);
  void tt_store_playout(uint64_t hash, int32_t score_rel);

  // Search position (one scratch board reused across the whole solve) and the
  // side-to-move state that make/unmake maintain alongside it.
  Board board_;
  const Dictionary* dict_ = nullptr;
  Rack racks_[2];
  int32_t scores_[2] = {0, 0};
  int scoreless_ = 0;
  int stm_ = 0;  // seat to move; 0 is the solving side at the root
  bool game_over_ = false;
  uint64_t board_hash_ = 0;

  std::vector<Frame> frames_;  // make/unmake stack, indexed by path length

  // Budget control.
  uint64_t nodes_ = 0;
  uint64_t budget_ = 0;
  bool aborting_ = false;

  std::vector<TTEntry> tt_;
  uint64_t tt_mask_ = 0;
  uint16_t tt_gen_ = 1;  // current generation; entries with gen != this are empty
};

}  // namespace scribblez
