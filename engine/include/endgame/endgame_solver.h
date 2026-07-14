#pragma once

#include "endgame/outplays.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

class Dictionary;
class LeaveOutplays;

// What a solve optimizes, given as EndgameSolver::solve's `objective`.
//
// kSpread searches the full window for the exact final spread. Exact spread
// play is also class-optimal (a positive spread IS a win), so this objective
// only "risks" the win/draw/loss class where its answer is an unproven
// estimate.
//
// kFirstWin pins the root window to (kFirstWinAlpha, kFirstWinBeta) around an
// even final spread, so the search only resolves the win/draw/loss class:
// proofs land far cheaper, and iterative deepening stops the moment the class
// is proven -- the break-out mode for self-play generation, where a proven
// class means the rest of the game is not worth computing. Among winning root
// moves it returns an arbitrary one, and a proven-loss best move is
// meaningless (every move loses).
//
// kLexicographic never trades the class for points: it first runs a kFirstWin
// pass to prove the class, then spends the remaining budget on a spread pass.
// With a proven win or draw, the spread pass's move is played only if it
// provably preserves the class (verified with a narrow-window probe when the
// spread pass is itself unproven; the proven-class move is the fallback). A
// proven loss makes every move class-equal, so the spread pass is pure defense
// (maximize the final spread of a lost game). When no class proof lands within
// budget, the spread pass's margin-maximizing answer is the class-robust
// fallback: margin is slack against estimate error.
enum class EndgameObjective : uint8_t { kSpread, kFirstWin, kLexicographic };

// The objective's CLI/config string form ("spread", "first-win",
// "lexicographic") and its inverse; parse_endgame_objective throws
// std::runtime_error on an unknown name.
// TODO(enum-strings): generate this pair with a framework like magic_enum
// instead of hand-maintaining the mapping.
const char* endgame_objective_name(EndgameObjective objective);
EndgameObjective parse_endgame_objective(const std::string& name);

// Result of a single endgame solve, from the perspective of the side to move
// at the solved position ("the solving side").
struct EndgameResult {
  // proven_class value when no class proof landed.
  static constexpr int kClassUnknown = 2;

  Move best;                         // best move for the solving side
  int32_t value = 0;                 // optimal final spread (solving score - opponent
                                     // score, after end-of-game adjustments); under
                                     // kFirstWin only its class (sign) is meaningful
  int depth_completed = 0;           // deepest fully-completed iterative-deepening depth
                                     // (0: the solve was declined up front or the budget
                                     // ran out inside the first iteration, so `best` is
                                     // only the statically best-estimated root move)
  uint64_t nodes = 0;                // negamax entries + greedy-playout plies spent
  bool proven = false;               // true iff `value` rests entirely on real game ends,
                                     // with no greedy-playout leaf in the lines that
                                     // determined it: then `value` is the exact
                                     // game-theoretic spread (under kFirstWin, a true
                                     // win/draw/loss verdict), not a depth-limited
                                     // estimate. Declined solves (depth_completed == 0)
                                     // are never proven.
  int proven_class = kClassUnknown;  // the position's proven game-theoretic
                                     // class for the solving side (+1 win,
                                     // 0 draw, -1 loss), or kClassUnknown; the
                                     // signal a self-play caller can break out
                                     // of the game on
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
// deepening stops the moment an iteration returns one; a proven narrow-window
// value is a true win/draw/loss verdict, so deepening stops once the class is
// settled. Proven-ness rides along every search result and every
// transposition-table entry, propagated through the negamax recursion (a node
// is proven only if the child searches its verdict rests on are all proven).
// What a solve does with proofs is the objective's business: see
// EndgameObjective.
//
// Opponent-outplay futility pruning cuts the mover's own moves at interior
// nodes. Each node knows the current out-play set of the side that will reply
// (see outplays.h): every entry is a play that would empty the replier's rack,
// still legal at its recorded score. A mover move m scoring g that leaves the
// replier a turn and places no tile in the halo of such an out-play scoring p
// lets the replier end the game at once, so
//   U(m) = s + g - p - 2*(pv - face(m))
// -- s the mover's spread at the node, pv its rack value, face(m) the face
// value of m's placed tiles -- is a sound upper bound on m's search value. The
// surviving out-play is just one replier option: out-plays the set is missing
// (kill-filter false drops; ones m newly enables) only lower m's true value,
// and halo conservatism only discards bounds, so the bound never overreaches
// downward. The best surviving out-play gives the tightest bound; a move with
// U(m) <= alpha is skipped (proven-grade, since the guaranteed line is
// terminal, so the skip never clears the node's proven bit), and U(m) also caps
// the move-ordering estimate so provably weak moves sink. Moves that end the
// game themselves -- an out-play, or a pass that trips the internal scoreless
// cap -- give the replier no turn and are never pruned. A test can disable the
// scheme (set_outplay_futility) to A/B that it changes no value or best move.
//
// The out-play sets are maintained incrementally rather than recomputed per
// node, since a per-node move generation for the replier's rack would cost as
// much as the search work it saves. The opponent's root set costs one extra
// move generation per solve; after that, making a move m by side s hands the
// child (a) for side s, the out-plays of m's leave -- read out of the node's
// own legal-play list by used-tile bucketing (LeaveOutplays), since any play of
// a rack subset is in that list -- and (b) for the other side, the parent's set
// filtered to the entries whose halo m does not touch. Halos are built on the
// board of the node that collected the entry and stay sound kill triggers as
// the board grows (see OutplayEntry). Greedy playouts neither consult nor
// maintain the sets.
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

  // Solve the position for the side holding my_rack (to move) under
  // `objective` (see EndgameObjective). Both racks must be the actual
  // remaining tiles (bag empty). scoreless_turns is the number of consecutive
  // zero-score turns already played in the real game.
  //
  // The search looks at most max_plies deep, and node_budget is a hard cap on
  // nodes spent across every pass the objective runs (exceeded by at most one
  // greedy playout's plies before the abort lands). A position with more root
  // moves than node_budget provably cannot complete a first iteration, so the
  // solve is declined immediately after root move generation rather than
  // burning the budget on a fraction of the root. A legal move is always
  // returned: the last completed iteration's best, else the best
  // fully-searched root move of the partial first iteration, else the
  // estimate-ordered top root move.
  EndgameResult solve(const Board& board, const Dictionary& dict, const Rack& my_rack,
                      const Rack& opp_rack, int my_score, int opp_score, int scoreless_turns,
                      uint64_t node_budget, int max_plies,
                      EndgameObjective objective = EndgameObjective::kSpread);

  // The first-win root window: a final spread >= kFirstWinBeta is a win,
  // <= kFirstWinAlpha a loss, 0 a draw.
  static constexpr int32_t kFirstWinAlpha = -1;
  static constexpr int32_t kFirstWinBeta = 1;

  // Invalidate every transposition-table entry (call between games; entries are
  // spread-rebased and thus reusable across turns within one game, but not
  // across games). O(1): it bumps the table's generation counter, and probes
  // treat entries from older generations as empty.
  void clear();

  // Enable or disable opponent-outplay futility pruning (on by default). Exists
  // so a test can A/B the two modes and assert the pruning never changes a
  // solve's value or best move; production always leaves it on.
  void set_outplay_futility(bool on) { outplay_futility_ = on; }

  // Enable or disable the proven-verdict deepening short-circuit (on by
  // default). Exists so tests and the benchmark can A/B how many nodes the
  // short-circuit saves; production always leaves it on.
  void set_proof_early_exit(bool on) { proof_early_exit_ = on; }

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

  // A candidate move with the value the search currently ranks it by: an
  // order_estimate() before its first root search, thereafter the value its
  // last root search returned.
  struct RankedMove {
    Move move;
    int32_t rank = 0;
  };
  static bool by_rank_desc(const RankedMove& a, const RankedMove& b) { return a.rank > b.rank; }

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
  // One full iterative-deepening pass over the already-prepared root at the
  // window (alpha, beta), sharing the solve-wide node budget and warm TT with
  // any pass run before it. Deepening stops on a proven verdict -- an exact
  // spread for the full window, a settled class for the first-win window
  // (`first_win` selects which exit test applies).
  EndgameResult run_iterative(int32_t alpha, int32_t beta, bool first_win,
                              std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                              int max_plies);
  // The kLexicographic driver over run_iterative passes; see EndgameObjective.
  EndgameResult solve_lexicographic(std::vector<RankedMove>& root_moves,
                                    const std::vector<Move>& plays, int max_plies);
  // True iff playing `m` at the root provably preserves game class `cls` for
  // the solving side: a narrow-window iterative probe of the child position,
  // sharing the solve budget. False when the proof does not land in budget.
  bool verify_move_class(const Move& m, int cls, const std::vector<Move>& plays, int max_plies);
  SearchResult run_root(int depth, int32_t alpha, int32_t beta, std::vector<RankedMove>& root_moves,
                        const std::vector<Move>& plays, Move* best_out);
  SearchResult negamax(int depth, int32_t alpha, int32_t beta, int ply);
  // Search one child at (alpha, beta) and return its value from the parent's
  // perspective (negated) with the child's proven bit, using a full window for
  // the first child and a scout-plus-conditional-re-search for the rest (PVS).
  SearchResult search_child(int child_depth, int32_t alpha, int32_t beta, int child_ply,
                            bool first);
  int32_t greedy_playout(uint64_t node_key, int ply);

  // Sound upper bound on the mover's value from playing `m` at the current node,
  // derived from the best out-play in `replier_outs` that `m` provably leaves
  // intact (see the class comment for the formula). Returns kInf when no
  // out-play survives `m` or `m` ends the game itself (the replier then never
  // gets a turn), so such moves are never pruned.
  int32_t outplay_futility_bound(const Move& m, const OutplaySet& replier_outs) const;

  // Derive the out-play sets the child reached by the mover's move `m` sees
  // (see the class comment), writing them into ply_sets_[child_ply] and
  // pointing cur_sets_ at them; restore_outplay_sets undoes the pointer swap.
  // `leave_outs` is the node's bucketing of its own play list. No-ops (and
  // returns cleanly) when the futility scheme is disabled.
  void push_outplay_sets(const Move& m, LeaveOutplays* leave_outs, int child_ply,
                         OutplaySet* saved[2]);
  void restore_outplay_sets(OutplaySet* const saved[2]);

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
  // `replier_outs` (when non-null) caps each estimate by the move's futility
  // bound, sinking provably weak moves.
  void order_moves(std::vector<RankedMove>& moves, const Move& tt_move, bool have_tt_move,
                   const OutplaySet* replier_outs) const;
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

  // Opponent-outplay futility pruning, on except when a test disables it to A/B
  // the two modes (see set_outplay_futility).
  bool outplay_futility_ = true;

  // The out-play sets each ply's children read their futility bounds from. A
  // node's mover writes its children's sets into ply_sets_[child ply] (one slot
  // per depth, so siblings reuse it) and points cur_sets_ there; cur_sets_[s]
  // is always seat s's current set, rooted at root_sets_ (the solving side's
  // slot stays empty: the root node itself is never futility-pruned).
  struct PlyOutplaySets {
    OutplaySet mover, other;
  };
  OutplaySet root_sets_[2];
  std::vector<PlyOutplaySets> ply_sets_;  // indexed by the child's ply
  OutplaySet* cur_sets_[2] = {nullptr, nullptr};

  bool proof_early_exit_ = true;
  std::vector<TTEntry> tt_;
  uint64_t tt_mask_ = 0;
  uint16_t tt_gen_ = 1;  // current generation; entries with gen != this are empty
};

}  // namespace scribblez
