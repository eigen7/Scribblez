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

// One bag-empty position to solve. `my_rack` belongs to the side to move, and
// scoreless_turns counts the consecutive zero-score turns already played.
struct EndgameState {
  const Dictionary* dict = nullptr;
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless_turns = 0;
};

// Result of one solve, from the perspective of the side to move at the solved
// position ("the solving side").
struct EndgameResult {
  // proven_class value when no class proof landed.
  static constexpr int kClassUnknown = 2;

  Move best;
  int32_t value = 0;        // optimal final spread; under kFirstWin only its sign is meaningful
  int depth_completed = 0;  // deepest fully-completed iterative-deepening depth, 0 if none
  uint64_t nodes = 0;       // negamax entries + greedy-playout plies spent
  bool proven = false;      // `value` is exact rather than a depth-limited estimate
  int proven_class = kClassUnknown;  // +1 win, 0 draw, -1 loss, or kClassUnknown

  // Proof-certificate line after `best`: the remaining moves of the game, both
  // sides alternating, ending it at the proven class. Present for every solve
  // that proves the class.
  std::vector<Move> continuation;
  // Nodes the certificate reconstruction spent; exempt from node_budget and so
  // tracked apart from `nodes`.
  uint64_t certificate_nodes = 0;
  // Logical move generations the solve performed. A memo hit counts as a real
  // generation, so this is a deterministic operation count a benchmark can
  // convert into modeled time.
  uint64_t movegens = 0;
};

// Exact/near-exact endgame solver for a bag-empty position with both racks
// known. A self-contained negamax search (alpha-beta + principal variation
// search + iterative deepening + a transposition table) with a greedy leaf
// playout, depending only on (board, dictionary, racks, scores) -- no equity
// model or leave tables.
//
// Scoreless turns are compressed relative to the full game rule (six
// consecutive zero-score turns end the game). Since a pass leaves the position
// unchanged, a chain of passes adds nothing to the search, so the solver ends
// the game after two consecutive scoreless turns, each side subtracting its own
// rack value. The caller's real count is rebased to 1 when positive; any play
// resets it to 0.
//
// A result is "proven" when its value rests entirely on real game ends, with no
// greedy-playout leaf in the lines that determined it: a proven full-window
// value is the exact game-theoretic spread, a proven narrow-window value a true
// win/draw/loss verdict. Iterative deepening stops as soon as a proof lands.
// Proven-ness propagates through the recursion and through the transposition
// table.
//
// Opponent-outplay futility pruning: each node knows the out-plays available to
// the side that will reply -- plays that would empty the replier's rack, still
// legal at their recorded score (see outplays.h). A mover move m scoring g that
// leaves such an out-play scoring p unblocked lets the replier end the game at
// once, so
//   U(m) = s + g - p - 2*(pv - face(m))
// -- s the mover's spread, pv its rack value, face(m) the face value of m's
// placed tiles -- is a sound upper bound on m's value. A move with
// U(m) <= alpha is skipped (proven-grade, since the guaranteed line is
// terminal), and U(m) also caps the move-ordering estimate. Moves that end the
// game themselves give the replier no turn and are never pruned. The root
// prunes only under the narrow first-win window; a full-window root needs every
// root move's exact value to re-order between iterations (see run_root).
//
// The out-play sets are maintained incrementally, since a per-node generation
// for the replier's rack would cost as much as the search work it saves. The
// opponent's root set costs one extra move generation per solve; thereafter a
// move m by side s hands the child (a) for side s, the out-plays of m's leave,
// read out of the node's own play list by used-tile bucketing (LeaveOutplays),
// and (b) for the other side, the parent's set filtered to entries whose halo m
// does not touch. Greedy playouts neither consult nor maintain the sets.
//
// Move generation is likewise incremental (PathMoveLists, on by default): a
// node's list is derived from the same side's list two plies up, carrying over
// lanes untouched by the two intervening moves and regenerating the rest. The
// derived list is byte-identical to a scratch generation, so only the cost of a
// "move generation" changes.
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
  // Renders a move against the board it is about to be played on.
  using MoveFormatter = std::function<std::string(const Board&, const Move&)>;

  // tt_log2_entries sizes the transposition table to 2^tt_log2_entries entries.
  explicit EndgameSolver(int tt_log2_entries = 16);

  // Solve-time configuration.
  struct Params {
    // Hard cap on nodes spent across every pass of a solve. A position with
    // more root moves than the budget is declined up front. The default is
    // tuned with `endgame_bench`; see docs/endgame_bench_results.md.
    uint64_t budget = 220;
    int plies = 25;  // iterative-deepening depth cap

    // When disabled, all provably winning moves are considered equally good,
    // as are all provably losing ones. When enabled, winning by A is a better
    // result than winning by B for A > B -- but the class is never traded for
    // points: a spread-maximizing move is played only when it provably
    // preserves a proven win or draw.
    bool spread_matters = false;

    // Register --<prefix>budget, --<prefix>plies, and
    // --<prefix>spread-matters, bound to this object's fields; the current
    // field values become the option defaults.
    void add_options(boost::program_options::options_description& desc,
                     const std::string& prefix = "");
  };

  // Solve `state` for the side holding its my_rack. A legal move is always
  // returned, even when the search is declined or aborted.
  EndgameResult solve(const EndgameState& state, const Params& params);

  // The first-win root window: a final spread >= kFirstWinBeta is a win,
  // <= kFirstWinAlpha a loss, 0 a draw.
  static constexpr int32_t kFirstWinAlpha = -1;
  static constexpr int32_t kFirstWinBeta = 1;

  // Invalidate every transposition-table entry. Must be called between games
  // (entries are reusable across turns within one game, but not across games).
  // O(1): it bumps the table's generation counter.
  void clear();

  // Enable a detailed trace of each solve to `os`, for debugging. Pass nullptr
  // to disable (the default).
  void set_trace(std::ostream* os, MoveFormatter fmt);

  // Search-feature toggles, all on by default. They exist so tests can A/B that
  // a feature changes no result and benchmarks can measure what it saves;
  // production always leaves them on.
  void set_outplay_futility(bool on) { outplay_futility_ = on; }  // futility pruning as a whole
  void set_root_futility(bool on) { root_futility_ = on; }        // its root-level leg
  void set_proof_early_exit(bool on) { proof_early_exit_ = on; }  // stop deepening on a proof
  void set_root_cutoff(bool on) { root_cutoff_ = on; }            // root beta cutoff
  void set_incremental_movegen(bool on) { incremental_movegen_ = on; }  // PathMoveLists

  // Enable or disable the move-generation memo (off by default): move lists are
  // cached by board+rack, trading memory for generation work. A benchmark
  // accelerator, not a production default -- a cache hit still counts as one
  // logical generation, so memo-on and memo-off solves are bit-identical
  // including movegens.
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

  // A transposition-table entry (32 bytes). The value is stored relative to the
  // node's spread, so it is reusable at any absolute score; the full hash
  // rejects index collisions, and entries from older generations read as empty
  // (see clear()).
  struct TTEntry {
    uint64_t hash = 0;
    Move best;
    int32_t score_rel = 0;
    uint8_t flag = kEmpty;
    uint8_t depth = 0;
    uint16_t gen = 0;
  };

  // A negamax return: the search value and whether it is proven.
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

  // Everything one make() changes, so unmake() can restore it.
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
  // One iterative-deepening pass over the already-prepared root at the window
  // (alpha, beta), sharing the solve-wide node budget and warm TT with any pass
  // run before it. `first_win` selects which proven-verdict exit test applies.
  EndgameResult run_iterative(int32_t alpha, int32_t beta, bool first_win,
                              std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                              int max_plies);
  // The spread_matters driver: a class pass, then a spread pass whose move is
  // played only when it provably preserves the proven class.
  EndgameResult solve_lexicographic(std::vector<RankedMove>& root_moves,
                                    const std::vector<Move>& plays, int max_plies);
  // True iff playing `m` at the root provably preserves game class `cls` for the
  // solving side. False when the proof does not land in budget.
  bool verify_move_class(const Move& m, int cls, const std::vector<Move>& plays, int max_plies);
  // Render `m` with the trace formatter against the solver's current board.
  std::string trace_move(const Move& m) const { return trace_fmt_(board_, m); }
  // Trace the replier's out-plays and, per root move, its futility bound.
  void trace_root_view(const std::vector<RankedMove>& root_moves);

  // Fill result.continuation with a proof-certificate line for result.best. The
  // class-critical side's moves come from fresh narrow-window proofs, searched
  // over the warm table outside the node budget; the doomed side plays the
  // greedy move, which cannot change the class. Cost lands in
  // result.certificate_nodes.
  void extract_continuation(EndgameResult& result);
  // The proven class-preserving move at walk ply `ply`, where `req_class` is the
  // required class from the mover's perspective. Returns false only if no proof
  // lands within kMaxPlayout depths.
  bool reprove_walk_move(int ply, int req_class, Move* out);
  SearchResult run_root(int depth, int32_t alpha, int32_t beta, bool first_win,
                        std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                        Move* best_out);
  SearchResult negamax(int depth, int32_t alpha, int32_t beta, int ply);
  // Search one child and return its value from the parent's perspective
  // (negated), using a full window for the first child and a scout-plus-
  // conditional-re-search for the rest (PVS).
  SearchResult search_child(int child_depth, int32_t alpha, int32_t beta, int child_ply,
                            bool first);
  int32_t greedy_playout(uint64_t node_key, int ply);

  // The legal move list for `rack` at the current position, `ply` deep down the
  // search path, counting one logical move generation. The returned reference
  // stays valid for the caller's use even across nested make/unmake.
  const std::vector<Move>& generate_moves(const Rack& rack, int ply);

  // Scratch move generation for `rack` on the current board, counting one
  // logical move generation. With the memo on it keys by board_hash_ mixed with
  // rack.bits() and returns a cached list on a hit. A 64-bit key collision would
  // return a wrong list -- the same accepted collision model as the
  // transposition table.
  const std::vector<Move>& generate_moves_scratch(const Rack& rack);

  // Sound upper bound on the mover's value from playing `m` (see the class
  // comment for the formula). Returns kInf when no out-play survives `m` or `m`
  // ends the game itself, so such moves are never pruned.
  int32_t outplay_futility_bound(const Move& m, const OutplaySet& replier_outs) const;

  // Derive the out-play sets the child reached by `m` sees, writing them into
  // ply_sets_[child_ply] and pointing cur_sets_ at them; restore_outplay_sets
  // undoes the pointer swap. `leave_outs` is the node's bucketing of its own
  // play list. A no-op when the futility scheme is disabled.
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

  // Logical move generations the current solve has performed, reset at solve()
  // entry (see EndgameResult::movegens).
  uint64_t movegens_ = 0;

  // Move-generation memo, keyed by board_hash_ mixed with the rack bits (see
  // set_movegen_memo). It survives clear(), its keys being board+rack content,
  // but the whole map is erased once it exceeds kMovegenMemoCap entries.
  static constexpr size_t kMovegenMemoCap = static_cast<size_t>(1) << 20;
  bool movegen_memo_ = false;
  std::unordered_map<uint64_t, std::vector<Move>> movegen_memo_map_;
  std::vector<Move> movegen_scratch_;  // return buffer when the memo is off

  // Incremental move-list maintenance and the per-ply lists it maintains.
  bool incremental_movegen_ = true;
  PathMoveLists path_lists_;

  bool outplay_futility_ = true;
  bool root_futility_ = true;

  // The out-play sets each ply's children read their futility bounds from. A
  // node's mover writes its children's sets into ply_sets_[child ply] (one slot
  // per depth, so siblings reuse it) and points cur_sets_ there; cur_sets_[s] is
  // always seat s's current set, rooted at root_sets_. The solving side's root
  // slot stays empty: the root's own pruning reads only the opponent's set.
  struct PlyOutplaySets {
    OutplaySet mover, other;
  };
  OutplaySet root_sets_[2];
  std::vector<PlyOutplaySets> ply_sets_;  // indexed by the child's ply
  OutplaySet* cur_sets_[2] = {nullptr, nullptr};

  bool proof_early_exit_ = true;
  bool root_cutoff_ = true;

  // Trace sink and move renderer; tracing is active iff trace_ is non-null
  // (set_trace installs a fallback renderer when none is given).
  std::ostream* trace_ = nullptr;
  MoveFormatter trace_fmt_;

  std::vector<TTEntry> tt_;
  uint64_t tt_mask_ = 0;
  uint16_t tt_gen_ = 1;  // current generation; entries with gen != this are empty
};

}  // namespace scribblez
