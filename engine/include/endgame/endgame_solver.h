#pragma once

#include "endgame/outplays.h"
#include "endgame/path_move_lists.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward-declared so Params::add_options() can register options without
// pulling boost::program_options into every consumer of this header.
namespace boost::program_options {
class options_description;
}

namespace scribblez {

class Dictionary;
class LeaveOutplays;

// One bag-empty position for EndgameSolver::solve: `my_rack` belongs to the
// side to move, both racks are the actual remaining tiles, and
// scoreless_turns counts the consecutive zero-score turns already played in
// the real game.
struct EndgameState {
  const Dictionary* dict = nullptr;  // lexicon the position is played under
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless_turns = 0;
};

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
  // The proof-certificate line after `best`: the remaining moves of the game,
  // both sides alternating, ending the game (under the solver's compressed
  // scoreless rule) at the proven class. Present for every solve that proves
  // the class: the class-critical side's moves come from fresh narrow-window
  // proofs at each position, so the line lands on the proven class by
  // construction (and a terminal check still gates it). The doomed side's
  // entries are proof-grade optimal-play stand-ins, not what the opposing
  // agent would have chosen.
  std::vector<Move> continuation;
  // Nodes the certificate reconstruction's re-searches spent. Tracked apart
  // from `nodes`: reconstruction runs after the search proper and is exempt
  // from node_budget (its cost is bounded in practice by the warm table).
  uint64_t certificate_nodes = 0;
  // The number of logical move generations the solve performed across search,
  // greedy playouts, and certificate reconstruction. Certificate
  // reconstruction runs inside solve(), so its generations land in this same
  // count. A move-generation memo hit counts the same as a real generation, so
  // the number reflects what an isolated, memo-less solve would compute -- a
  // deterministic operation count a benchmark can convert into modeled time.
  uint64_t movegens = 0;
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
// What a solve does with proofs is Params::spread_matters' business.
//
// Opponent-outplay futility pruning cuts the mover's own moves at interior
// nodes and -- under the narrow first-win window -- at the root. Each node
// knows the current out-play set of the side that will reply
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
// cap -- give the replier no turn and are never pruned. At the root the same
// bound test runs only under the narrow first-win window, where a pruned
// move's proven terminal bound settles its contribution to the class verdict;
// a full-window root never prunes, because the re-ordering between deepening
// iterations needs every root move's exact value (see run_root). A test can
// disable the scheme (set_outplay_futility) to A/B that it changes no value or
// best move, and the root leg separately (set_root_futility) to A/B that it
// changes no class verdict.
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
// Move generation itself is also incremental (PathMoveLists, on by default):
// both sides' full legal-play lists are generated once on the root board, and
// every deeper node's list is derived from the same side's list two plies up
// -- lanes untouched by the two intervening moves carry over verbatim (subset-
// filtered against the mover's shrunken rack), touched lanes are regenerated
// via MoveGenerator::generate_lane. The derived list is byte-identical to a
// scratch generation, so solves are bit-for-bit the same either way; only the
// cost of a "move generation" changes. A test can disable the scheme
// (set_incremental_movegen) to A/B exactly that.
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

  // Solve-time configuration.
  //
  // spread_matters selects what a solve optimizes. false: resolve only the
  // win/draw/loss class (narrow root window; iterative deepening stops the
  // moment the class is proven, and among winning moves an arbitrary one is
  // returned) -- the self-play break-out setting, where a proven class means
  // the rest of the game is not worth computing. true: never trade the class
  // for points -- a class pass capped at half the budget, then a spread pass
  // on the remainder whose move is played only when it provably preserves a
  // proven win or draw (a proven spread pass implies that; an unproven one is
  // checked with a narrow-window probe, falling back to the proven-class
  // move); a proven loss makes every move class-equal, so the spread pass is
  // pure defense; and with no class proof in budget, margin-maximizing play
  // is the class-robust fallback (margin is slack against estimate error).
  struct Params {
    // Hard cap on nodes spent across every pass a solve runs (exceeded by at
    // most one greedy playout's plies before the abort lands). A position
    // with more root moves than the budget provably cannot complete a first
    // iteration and is declined up front. The default is tuned with
    // `endgame_bench` on NWL23 as the largest budget whose endgame-vs-endgame
    // games stay within ~2x the plain HastyBot-vs-HastyBot game time; see
    // docs/endgame_bench_results.md.
    uint64_t budget = 220;
    int plies = 25;  // iterative-deepening depth cap
    bool spread_matters = false;

    // Register --<prefix>budget, --<prefix>plies, and
    // --<prefix>spread-matters, bound to this object's fields; the current
    // field values become the option defaults.
    void add_options(boost::program_options::options_description& desc,
                     const std::string& prefix = "");
  };

  // Solve `state` for the side holding its my_rack. A legal move is always
  // returned: the last completed iteration's best, else the best
  // fully-searched root move of the partial first iteration, else the
  // estimate-ordered top root move.
  EndgameResult solve(const EndgameState& state, const Params& params);

  // The first-win root window: a final spread >= kFirstWinBeta is a win,
  // <= kFirstWinAlpha a loss, 0 a draw.
  static constexpr int32_t kFirstWinAlpha = -1;
  static constexpr int32_t kFirstWinBeta = 1;

  // Invalidate every transposition-table entry (call between games; entries are
  // spread-rebased and thus reusable across turns within one game, but not
  // across games). O(1): it bumps the table's generation counter, and probes
  // treat entries from older generations as empty.
  void clear();

  // Stream a human-readable trace of each solve to `os`: the position summary,
  // the replier's out-play set, every root move's futility bound (the
  // block-or-outscore view), each iteration's verdict, and the certificate
  // walk. `fmt` renders a move against the board it is about to be played on
  // (the endgame tool binds GCG notation). Pass nullptr to disable (the
  // default).
  void set_trace(std::ostream* os, std::function<std::string(const Board&, const Move&)> fmt);

  // Enable or disable opponent-outplay futility pruning (on by default). Exists
  // so a test can A/B the two modes and assert the pruning never changes a
  // solve's value or best move; production always leaves it on.
  void set_outplay_futility(bool on) { outplay_futility_ = on; }

  // Enable or disable root-level outplay futility pruning (on by default):
  // under the narrow first-win window, a root move whose futility bound cannot
  // beat alpha is skipped without search, exactly as at an interior node. Full
  // windows are unaffected (they never prune the root), as is everything when
  // outplay futility itself is off. Exists so a test can A/B the two modes and
  // assert the pruning changes no proven class; production always leaves it on.
  void set_root_futility(bool on) { root_futility_ = on; }

  // Enable or disable the proven-verdict deepening short-circuit (on by
  // default). Exists so tests and the benchmark can A/B how many nodes the
  // short-circuit saves; production always leaves it on.
  void set_proof_early_exit(bool on) { proof_early_exit_ = on; }

  // Enable or disable the root beta cutoff (on by default). Exists so a test
  // can A/B how many nodes the cutoff saves; production always leaves it on.
  // The cutoff only ever fires under the narrow first-win window (the full
  // window's +infinity beta is unreachable), where a root fail-high already
  // settles the class, so disabling it changes no solve's value or best move.
  void set_root_cutoff(bool on) { root_cutoff_ = on; }

  // Enable or disable incremental move-list maintenance (on by default): node
  // move lists are derived from the lists two plies up via PathMoveLists
  // instead of generated from scratch, with byte-identical results. Exists so
  // a test can A/B the two modes and assert bit-identical solves, and so the
  // benchmark can measure the speedup; production always leaves it on.
  void set_incremental_movegen(bool on) { incremental_movegen_ = on; }

  // Enable or disable the move-generation memo (off by default). When on, every
  // move list the solver requests is cached by board+rack and reused on a
  // repeat, trading memory for generation work. It is a measurement-tool
  // accelerator, not a production default: the logical `movegens` count is
  // unchanged (a cache hit still counts as one generation), so a memo-on and a
  // memo-off solve return bit-identical results including movegens. Exists so a
  // benchmark can accelerate deterministic-operation-count runs.
  void set_movegen_memo(bool on) { movegen_memo_ = on; }

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
  // The spread_matters driver over run_iterative passes; see Params.
  EndgameResult solve_lexicographic(std::vector<RankedMove>& root_moves,
                                    const std::vector<Move>& plays, int max_plies);
  // True iff playing `m` at the root provably preserves game class `cls` for
  // the solving side: a narrow-window iterative probe of the child position,
  // sharing the solve budget. False when the proof does not land in budget.
  bool verify_move_class(const Move& m, int cls, const std::vector<Move>& plays, int max_plies);
  // Render `m` with the trace formatter against the solver's current board.
  std::string trace_move(const Move& m) const { return trace_fmt_(board_, m); }
  // The root block-or-outscore view: the replier's out-plays and, per root
  // move, the futility bound and the strongest out-play it fails to block.
  void trace_root_view(const std::vector<RankedMove>& root_moves);

  // Fill result.continuation with a proof-certificate line for result.best:
  // at every turn of a side that must preserve the proven class (the winner's
  // turns; both sides' in a draw) the move comes from a fresh narrow-window
  // proof of the current position, re-searched over the warm table outside the
  // node budget; the doomed side's turns take the greedy playout move, which
  // cannot change the class. Succeeds for every proven class; the terminal
  // class check is kept as a hard gate. Reconstruction cost lands in
  // result.certificate_nodes.
  void extract_continuation(EndgameResult& result);
  // The proven class-preserving move for the side to move at the current walk
  // position (`req_class` is the class from the mover's perspective), from an
  // iterative narrow-window proof rooted at walk ply `ply`. Returns false only
  // if no proof lands within kMaxPlayout depths (impossible for a true class,
  // kept as the never-wrong gate).
  bool reprove_walk_move(int ply, int req_class, Move* out);
  SearchResult run_root(int depth, int32_t alpha, int32_t beta, bool first_win,
                        std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                        Move* best_out);
  SearchResult negamax(int depth, int32_t alpha, int32_t beta, int ply);
  // Search one child at (alpha, beta) and return its value from the parent's
  // perspective (negated) with the child's proven bit, using a full window for
  // the first child and a scout-plus-conditional-re-search for the rest (PVS).
  SearchResult search_child(int child_depth, int32_t alpha, int32_t beta, int child_ply,
                            bool first);
  int32_t greedy_playout(uint64_t node_key, int ply);

  // Return the legal move list for `rack` at the current position, `ply` deep
  // down the search path, counting one logical move generation (movegens_).
  // With incremental maintenance on this reads PathMoveLists (byte-identical
  // to a scratch generation); otherwise it defers to generate_moves_scratch.
  // The returned reference stays valid while the caller uses it even across
  // nested make/unmake: deeper plies materialize into their own slots, and
  // the scratch buffer is overwritten only by a later same-mode call, never
  // during the current caller's use.
  const std::vector<Move>& generate_moves(const Rack& rack, int ply);

  // Scratch move generation for `rack` on the current board, counting one
  // logical move generation (movegens_). With the memo off it generates fresh
  // into a scratch buffer; with the memo on it keys by the current board_hash_
  // mixed with rack.bits() and returns a cached list on a hit, else generates,
  // stores, and returns. A 64-bit key collision with matching content would
  // return a wrong move list -- the same accepted collision model as the
  // transposition table -- but the map stores the full key, so its bucket
  // (index) collisions are resolved by key equality and only a genuine hash
  // collision can misfire.
  const std::vector<Move>& generate_moves_scratch(const Rack& rack);

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

  // Logical move generations the current solve has performed (see
  // EndgameResult::movegens). Reset at solve() entry, copied into the result at
  // the end; a memo hit increments it just as a real generation does.
  uint64_t movegens_ = 0;

  // Move-generation memo, off by default (see set_movegen_memo). Keyed by the
  // full 64-bit key mixing board_hash_ with the rack bits. It survives clear()
  // -- its keys are board+rack content, independent of scores and table
  // generations -- but is capped: once it exceeds kMovegenMemoCap entries the
  // whole map is erased before the next insert (a crude bound; the memo is a
  // measurement accelerator, not a production default).
  static constexpr size_t kMovegenMemoCap = static_cast<size_t>(1) << 20;
  bool movegen_memo_ = false;
  std::unordered_map<uint64_t, std::vector<Move>> movegen_memo_map_;
  std::vector<Move> movegen_scratch_;  // return buffer when the memo is off

  // Incremental move-list maintenance, on except when a test or benchmark
  // disables it to A/B the two modes (see set_incremental_movegen), and the
  // per-ply lists it maintains.
  bool incremental_movegen_ = true;
  PathMoveLists path_lists_;

  // Opponent-outplay futility pruning, on except when a test disables it to A/B
  // the two modes (see set_outplay_futility).
  bool outplay_futility_ = true;

  // Root-level leg of the futility scheme, on except when a test disables it to
  // A/B the two modes (see set_root_futility).
  bool root_futility_ = true;

  // The out-play sets each ply's children read their futility bounds from. A
  // node's mover writes its children's sets into ply_sets_[child ply] (one slot
  // per depth, so siblings reuse it) and points cur_sets_ there; cur_sets_[s]
  // is always seat s's current set, rooted at root_sets_ (the solving side's
  // slot stays empty: the root's own pruning reads only the opponent's set,
  // and every deeper node reads the per-ply sets push_outplay_sets installs).
  struct PlyOutplaySets {
    OutplaySet mover, other;
  };
  OutplaySet root_sets_[2];
  std::vector<PlyOutplaySets> ply_sets_;  // indexed by the child's ply
  OutplaySet* cur_sets_[2] = {nullptr, nullptr};

  bool proof_early_exit_ = true;

  // Root beta cutoff, on except when a test disables it to measure the nodes it
  // saves (see set_root_cutoff).
  bool root_cutoff_ = true;

  // Trace sink and move renderer; tracing is active iff trace_ is non-null
  // (set_trace installs a fallback renderer when none is given).
  std::ostream* trace_ = nullptr;
  std::function<std::string(const Board&, const Move&)> trace_fmt_;

  std::vector<TTEntry> tt_;
  uint64_t tt_mask_ = 0;
  uint16_t tt_gen_ = 1;  // current generation; entries with gen != this are empty
};

}  // namespace scribblez
