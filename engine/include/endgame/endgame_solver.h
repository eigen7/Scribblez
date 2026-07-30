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
#include <vector>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

class Dictionary;
class LeaveOutplays;

// A position to solve, from the perspective of the side holding `my_rack`.
// The bag must be empty, which is what makes both racks known.
struct EndgameState {
  const Dictionary* dict = nullptr;
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless_turns = 0;  // consecutive zero-score turns already played
};

// From the perspective of the side that was to move.
struct EndgameResult {
  // proven_class value when no class proof landed.
  static constexpr int kClassUnknown = 2;

  Move best;
  int32_t value = 0;        // final spread under optimal play; only its sign is
                            // meaningful when the solve optimized class alone
  int depth_completed = 0;  // deepest fully-completed search depth; 0 if the budget
                            // bought no complete iteration, leaving `best` a guess
  uint64_t nodes = 0;
  bool proven = false;               // `value` is exact rather than a depth-limited estimate
  int proven_class = kClassUnknown;  // +1 win, 0 draw, -1 loss, or kClassUnknown

  // A certificate for a proven class: the remaining moves of the game, both
  // sides alternating, ending it at that class. Empty when nothing was proven.
  std::vector<Move> continuation;
  uint64_t certificate_nodes = 0;  // nodes spent building `continuation`, which
                                   // is exempt from the node budget
  uint64_t movegens = 0;           // move generations the solve performed -- a
                                   // machine-independent cost measure for benchmarks
};

// The Scrabble endgame -- the phase after the bag empties -- is a game of
// perfect information, so the position has an exact game-theoretic value.
// EndgameSolver searches for it with alpha-beta, consulting nothing but the
// board, the lexicon, the racks, and the scores: no equity model, no leave
// tables, no belief over unseen tiles.
//
// A solve is bounded by a node budget rather than run to exhaustion, so its
// answer is either exact ("proven": every line it rests on reached a real game
// end) or an estimate from a search that ran out of room. A legal move always
// comes back, even from a budget too small to search at all, and where a proof
// lands the solver also returns a certificate: the line of play carrying the
// position from `best` to a game end at the proven class.
//
// The solver ends a game after two consecutive scoreless turns rather than the
// official six. A pass leaves the position unchanged, so a longer chain of them
// only reaches the same stalemate deeper down the tree.
//
// Speed comes from iterative deepening, a transposition table, the out-play
// futility pruning of outplays.h, and the incremental move-list maintenance of
// path_move_lists.h.
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
  // The board passed is the one the move is about to be played on.
  using MoveFormatter = std::function<std::string(const Board&, const Move&)>;

  // The transposition table takes 2^tt_log2_entries entries.
  explicit EndgameSolver(int tt_log2_entries = 16);

  struct Params {
    // Hard cap on the nodes one solve may spend. This is the endgame's
    // strength-for-throughput dial, and picking a value is a judgement rather
    // than an optimum: docs/endgame_bench_results.md plots both curves against
    // the start-of-endgame margin and is the place to look before changing it.
    uint64_t budget = 400;
    int plies = 25;  // iterative-deepening depth cap

    // When disabled, the solve stops at the class proof: all provably winning
    // moves are equally good, as are all provably losing ones. When enabled,
    // winning by more is better -- though never at the cost of the win: the
    // class comes first, and spread only breaks ties within it.
    //
    // Either way, a solve that fails to prove the class falls back to a
    // spread-maximizing pass. A narrow-window search that proves nothing ranks
    // its moves by a bound rather than by points, and deepening only reshuffles
    // them, so its move is no basis for play against an imperfect opponent.
    bool spread_matters = false;

    // Register the options under `prefix`, bound to this object's fields, whose
    // current values become the defaults.
    void add_options(boost::program_options::options_description& desc,
                     const std::string& prefix = "");
  };

  EndgameResult solve(const EndgameState& state, const Params& params);

  // The window a class-only solve searches: a final spread >= kFirstWinBeta is a
  // win, <= kFirstWinAlpha a loss, 0 a draw.
  static constexpr int32_t kFirstWinAlpha = -1;
  static constexpr int32_t kFirstWinBeta = 1;

  // Must be called between games; within a game, every turn reuses the table.
  void clear();

  // Trace each solve to `os`, or nullptr to stop (the default).
  void set_trace(std::ostream* os, MoveFormatter fmt);

  // Switches for individual search features, all on. Production never touches
  // them; they let a test assert that a feature changes no result, and a
  // benchmark measure what it saves.
  void set_outplay_futility(bool on) { outplay_futility_ = on; }
  void set_root_futility(bool on) { root_futility_ = on; }
  void set_proof_early_exit(bool on) { proof_early_exit_ = on; }
  void set_root_cutoff(bool on) { root_cutoff_ = on; }
  void set_incremental_movegen(bool on) { incremental_movegen_ = on; }

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

  // A transposition-table entry, packed into 32 bytes. `score_rel` is relative
  // to the node's spread, so an entry is reusable at any absolute score, and
  // `hash` is stored in full so an index collision can be rejected.
  struct TTEntry {
    uint64_t hash = 0;
    Move best;
    int32_t score_rel = 0;
    uint8_t flag = kEmpty;
    uint8_t depth = 0;
    uint16_t gen = 0;  // generation that wrote it; older ones read as empty
  };

  struct SearchResult {
    int32_t value = 0;
    bool proven = false;
  };

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
  // Deepen over the prepared root until the window (alpha, beta) is resolved,
  // the budget runs out, or `max_plies` is reached.
  EndgameResult run_iterative(int32_t alpha, int32_t beta, bool first_win,
                              std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                              int max_plies);
  // Prove the class on half the budget, then spend the rest: on the spread when
  // `refine_spread` (Params::spread_matters), or, when the class proof failed,
  // on a spread pass whose move is at least chosen for points.
  EndgameResult solve_class_first(std::vector<RankedMove>& root_moves,
                                  const std::vector<Move>& plays, int max_plies,
                                  bool refine_spread);
  // Whether playing `m` at the root provably preserves game class `cls`. False
  // when no proof lands in the remaining budget.
  bool verify_move_class(const Move& m, int cls, const std::vector<Move>& plays, int max_plies);
  std::string trace_move(const Move& m) const { return trace_fmt_(board_, m); }
  // Trace the opponent's out-plays and what each root move does about them.
  void trace_root_view(const std::vector<RankedMove>& root_moves);

  // Fill result.continuation with a certificate for result.best, re-searching
  // the line over the warm table. Runs outside the node budget, charging its
  // cost to result.certificate_nodes.
  void extract_continuation(EndgameResult& result);
  // The move that preserves class `req_class` for the side to move at walk ply
  // `ply`. False if no proof of one lands.
  bool reprove_walk_move(int ply, int req_class, Move* out);
  SearchResult run_root(int depth, int32_t alpha, int32_t beta, bool first_win,
                        std::vector<RankedMove>& root_moves, const std::vector<Move>& plays,
                        Move* best_out);
  SearchResult negamax(int depth, int32_t alpha, int32_t beta, int ply);
  // One child's value, from the parent's perspective.
  SearchResult search_child(int child_depth, int32_t alpha, int32_t beta, int child_ply,
                            bool first);
  // The spread from playing the position out greedily -- what a search that has
  // run out of depth falls back on.
  int32_t greedy_playout(uint64_t node_key, int ply);

  // The legal moves for `rack` at the current position, `ply` deep down the
  // search path. The reference stays valid across nested make/unmake and nested
  // generate_moves calls.
  const std::vector<Move>& generate_moves(const Rack& rack, int ply);
  // Generated straight from the board, bypassing the path lists: for the root,
  // whose lists are what this seeds, and for the incremental A/B. The reference
  // is invalidated by the next call, so a caller holding a list across further
  // generation keeps its own copy.
  const std::vector<Move>& generate_moves_scratch(const Rack& rack);

  // A sound upper bound on what the mover can get out of playing `m`, derived
  // from the best out-play in `replier_outs` that `m` leaves intact. kInf when
  // no out-play bounds `m`.
  int32_t outplay_futility_bound(const Move& m, const OutplaySet& replier_outs) const;

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
  // Sort `moves` best-first. `replier_outs`, when given, sinks the moves its
  // futility bound proves weak.
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

  // The search position -- one scratch board reused across the whole solve --
  // and the side-to-move state make/unmake maintain alongside it.
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

  uint64_t movegens_ = 0;  // see EndgameResult::movegens

  std::vector<Move> scratch_moves_;  // generate_moves_scratch's return buffer

  bool incremental_movegen_ = true;
  PathMoveLists path_lists_;

  bool outplay_futility_ = true;
  bool root_futility_ = true;
  OutplaySetStack outplay_sets_;

  bool proof_early_exit_ = true;
  bool root_cutoff_ = true;

  // Tracing is active iff trace_ is non-null; set_trace installs a fallback
  // renderer when given none.
  std::ostream* trace_ = nullptr;
  MoveFormatter trace_fmt_;

  std::vector<TTEntry> tt_;
  uint64_t tt_mask_ = 0;
  uint16_t tt_gen_ = 1;  // current generation; entries with gen != this are empty
};

}  // namespace scribblez
