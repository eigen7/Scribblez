#include "endgame/endgame_solver.h"

#include "agent/macondo_bot.h"
#include "endgame/outplays.h"
#include "game/glyph.h"
#include "game/movegen.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "util/math.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace scribblez {

namespace {

// Stands in for +/- infinity in the alpha-beta window: far above any real
// endgame spread, yet small enough not to overflow on negation.
constexpr int32_t kInf = 1'000'000;

// Upper bound on plies a single greedy leaf playout can add past the search
// depth. The combined racks hold at most 2*RACK_SIZE tiles, each play removes
// at least one, and two consecutive passes end the game, so a playout is always
// far shorter than this.
constexpr int kMaxPlayout = 40;

// One key per (square, letter, is-blank) placement, plus one per internal
// scoreless count, all derived from a fixed seed.
struct ZobristTable {
  std::array<std::array<std::array<uint64_t, 2>, 26>, BOARD_SIZE * BOARD_SIZE> square;
  std::array<uint64_t, 3> scoreless;
};

ZobristTable build_zobrist() {
  ZobristTable z;
  uint64_t s = 0xE3D1F0A2B4C6D8E9ULL;
  for (auto& sq : z.square)
    for (auto& letter : sq)
      for (auto& blank : letter) blank = util::splitmix64(++s);
  for (auto& v : z.scoreless) v = util::splitmix64(++s);
  return z;
}

// In the first-win window a proven result settles the win/draw/loss class iff
// its value is a proven win bound (>= kFirstWinBeta), a proven loss bound
// (<= kFirstWinAlpha), or a proven draw (0). A proven verdict in this window is
// always one of these, so this holds whenever the result is proven.
bool settles_first_win_class(int32_t value) {
  return value >= EndgameSolver::kFirstWinBeta || value <= EndgameSolver::kFirstWinAlpha ||
         value == 0;
}

// +1 win, 0 draw, -1 loss.
int class_of(int32_t value) { return (value > 0) - (value < 0); }

const ZobristTable& zobrist() {
  static const ZobristTable table = build_zobrist();
  return table;
}

}  // namespace

void EndgameSolver::Params::add_options(boost::program_options::options_description& desc,
                                        const std::string& prefix) {
  namespace po = boost::program_options;
  desc.add_options()  //
    ((prefix + "budget").c_str(), po::value<uint64_t>(&budget)->default_value(budget),
     "per-solve node budget (0 disables solving)")  //
    ((prefix + "plies").c_str(), po::value<int>(&plies)->default_value(plies),
     "iterative-deepening depth cap")  //
    ((prefix + "spread-matters").c_str(),
     po::value<bool>(&spread_matters)->default_value(spread_matters),
     "false: resolve only the win/draw/loss class and stop at its proof (the self-play "
     "break-out setting); true: prove the class first, then maximize spread without ever "
     "trading the class for points. Both fall back to a spread pass when the class proof "
     "fails");
}

EndgameSolver::EndgameSolver(int tt_log2_entries)
    : tt_(size_t(1) << tt_log2_entries), tt_mask_((uint64_t(1) << tt_log2_entries) - 1) {
  static_assert(sizeof(TTEntry) == 32, "transposition-table entry should pack into 32 bytes");
}

void EndgameSolver::clear() {
  ++tt_gen_;
  if (tt_gen_ == 0) {
    // Generation counter wrapped: entries written 2^16 generations ago would
    // read as current, so this one time the table really is wiped.
    std::fill(tt_.begin(), tt_.end(), TTEntry{});
    tt_gen_ = 1;
  }
}

uint64_t EndgameSolver::compute_board_hash() const {
  const ZobristTable& z = zobrist();
  uint64_t h = 0;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board_.at(r, c);
      if (!g.has_letter()) continue;
      h ^= z.square[r * BOARD_SIZE + c][g.letter().index()][g.is_blank() ? 1 : 0];
    }
  }
  return h;
}

void EndgameSolver::xor_play_hash(const Move& move) {
  const ZobristTable& z = zobrist();
  const bool horizontal = move.horizontal();
  const int start = move.start();
  uint16_t mask = move.square_mask();
  int gi = 0;
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    const Glyph g = move.glyph(gi++);
    board_hash_ ^= z.square[r * BOARD_SIZE + c][g.letter().index()][g.is_blank() ? 1 : 0];
  }
}

uint64_t EndgameSolver::node_hash() const {
  const ZobristTable& z = zobrist();
  uint64_t h = board_hash_;
  // The mover's and opponent's racks enter asymmetrically, so a physical
  // position hashes identically no matter which seat a solve was rooted at --
  // the property that lets both seats of one game share a table. Stored
  // values are relative to the mover's spread, so they are equally
  // seat-agnostic. (Equal racks make the two mover assignments collide, but
  // those are true transpositions: same board, same racks, same
  // mover-relative value.)
  h ^= util::splitmix64(racks_[stm_].bits());
  h ^= util::splitmix64(~racks_[1 - stm_].bits());
  h ^= z.scoreless[scoreless_];
  return h;
}

int EndgameSolver::placed_face_value(const Move& move) const {
  int total = 0;
  for (int i = 0; i < move.num_glyphs(); ++i) total += move.glyph(i).value();
  return total;
}

void EndgameSolver::make(const Move& move, int ply) {
  if (incremental_movegen_) path_lists_.on_make(ply, move);
  Frame& f = frames_[ply];
  f.scores[0] = scores_[0];
  f.scores[1] = scores_[1];
  f.scoreless = scoreless_;
  f.game_over = game_over_;
  f.mover = stm_;
  f.move = move;

  const int mover = stm_;
  const int opp = 1 - mover;
  if (move.type() == MoveType::PLAY) {
    xor_play_hash(move);
    board_.apply(move, &f.board_undo);
    for (int i = 0; i < move.num_glyphs(); ++i) racks_[mover].remove(move.glyph(i).rack_tile());
    scores_[mover] += move.score();
    scoreless_ = 0;  // any play resets the scoreless run
    if (racks_[mover].empty()) {
      // Out: bag is empty, so emptying the rack ends the game with a bonus of
      // twice the opponent's remaining tile value.
      scores_[mover] += 2 * racks_[opp].point_value();
      game_over_ = true;
    }
  } else {
    board_.apply(move, &f.board_undo);  // records the caches-valid flag only
    ++scoreless_;
    if (scoreless_ >= 2) {
      // Internal stalemate cap: each side subtracts its own remaining tiles.
      scores_[mover] -= racks_[mover].point_value();
      scores_[opp] -= racks_[opp].point_value();
      game_over_ = true;
    }
  }
  stm_ = opp;
}

void EndgameSolver::unmake(int ply) {
  Frame& f = frames_[ply];
  stm_ = f.mover;
  if (f.move.type() == MoveType::PLAY) {
    for (int i = 0; i < f.move.num_glyphs(); ++i) racks_[stm_].add(f.move.glyph(i).rack_tile());
    xor_play_hash(f.move);
  }
  board_.unapply(f.board_undo);
  scores_[0] = f.scores[0];
  scores_[1] = f.scores[1];
  scoreless_ = f.scoreless;
  game_over_ = f.game_over;
}

int32_t EndgameSolver::order_estimate(const Move& move, const Move& tt_move,
                                      bool have_tt_move) const {
  const int mover = stm_;
  int32_t est;
  if (move.type() == MoveType::PLAY && move.num_glyphs() == racks_[mover].size()) {
    // Out-play: score plus the endgame bonus, floated to the top.
    est = move.score() + 2 * racks_[1 - mover].point_value() + (1 << 27);
  } else if (move.type() == MoveType::PLAY) {
    est = int32_t(move.score()) - placed_face_value(move);
  } else {
    est = 0;  // PASS
  }
  if (have_tt_move && move == tt_move) est += (1 << 28);
  if (move.type() == MoveType::PASS && scoreless_ > 0) est += (1 << 29);
  return est;
}

void EndgameSolver::order_moves(std::vector<RankedMove>& moves, const Move& tt_move,
                                bool have_tt_move, const OutplaySet* replier_outs) const {
  for (RankedMove& m : moves) {
    m.rank = order_estimate(m.move, tt_move, have_tt_move);
    if (replier_outs == nullptr) continue;
    // A finite futility bound is a proven cap on the move's value. Rebasing it
    // to the current spread puts it in the same points-denominated units as the
    // estimates, and taking the min sinks provably weak moves -- including a
    // stale TT move whose payoff a replier out-play now caps.
    const int32_t u = outplay_futility_bound(m.move, *replier_outs);
    if (u < kInf) m.rank = std::min(m.rank, u - spread_stm());
  }
  std::sort(moves.begin(), moves.end(), by_rank_desc);
}

double EndgameSolver::playout_adjusted(const Move& move) const {
  const int mover = stm_;
  const int placed = placed_face_value(move);
  const bool empties = move.num_glyphs() == racks_[mover].size();
  const int rack_pv = racks_[mover].point_value();
  const int term = empties ? 2 * racks_[1 - mover].point_value() : -2 * (rack_pv - placed) - 10;
  return double(move.score()) + term;
}

const Move& EndgameSolver::greedy_pick(const std::vector<Move>& plays) const {
  size_t best = 0;
  double best_adj = playout_adjusted(plays[0]);
  for (size_t i = 1; i < plays.size(); ++i) {
    const double adj = playout_adjusted(plays[i]);
    if (hasty_move_better(adj, plays[i], best_adj, plays[best])) {
      best = i;
      best_adj = adj;
    }
  }
  return plays[best];
}

EndgameSolver::TTEntry* EndgameSolver::tt_probe(uint64_t hash) {
  TTEntry& e = tt_[hash & tt_mask_];
  if (e.gen == tt_gen_ && e.flag != kEmpty && e.hash == hash) return &e;
  return nullptr;
}

void EndgameSolver::tt_store(uint64_t hash, int32_t score_rel, uint8_t bound, bool proven,
                             const Move& best, int depth) {
  TTEntry& e = tt_[hash & tt_mask_];
  e.hash = hash;
  e.best = best;
  e.score_rel = score_rel;
  e.flag = tt_pack(bound, proven);
  e.depth = uint8_t(depth);
  e.gen = tt_gen_;
}

void EndgameSolver::tt_store_playout(uint64_t hash, int32_t score_rel) {
  TTEntry& e = tt_[hash & tt_mask_];
  // Never overwrite a current-generation real (deeper) search result with a
  // depth-0 playout value.
  if (e.gen == tt_gen_ && e.flag != kEmpty && e.depth > 0) return;
  e.hash = hash;
  e.best = Move::pass();  // placeholder: depth-0 entries never supply a TT move
  e.score_rel = score_rel;
  e.flag = kExact;
  e.depth = 0;
  e.gen = tt_gen_;
}

const std::vector<Move>& EndgameSolver::generate_moves(const Rack& rack, int ply) {
  if (!incremental_movegen_) return generate_moves_scratch(rack);
  ++movegens_;
  return path_lists_.moves_at(ply, rack);
}

const std::vector<Move>& EndgameSolver::generate_moves_scratch(const Rack& rack) {
  ++movegens_;
  scratch_moves_ = MoveGenerator(board_, *dict_).generate(rack);
  return scratch_moves_;
}

int32_t EndgameSolver::greedy_playout(uint64_t node_key, int ply) {
  const int solving_seat = stm_;
  const int32_t node_spread = spread_stm();
  int made = 0;
  while (!game_over_ && made < kMaxPlayout) {
    const std::vector<Move>& plays = generate_moves(racks_[stm_], ply + made);
    const Move chosen = plays.empty() ? Move::pass() : greedy_pick(plays);
    ++nodes_;
    make(chosen, ply + made);
    ++made;
  }

  int32_t value = scores_[solving_seat] - scores_[1 - solving_seat];
  if (!game_over_) {
    // Ran into the playout cap without a natural end: apply the stalemate-style
    // rack adjustment so the leaf value is still a plausible final spread.
    value -= racks_[solving_seat].point_value();
    value += racks_[1 - solving_seat].point_value();
  }
  for (int d = made - 1; d >= 0; --d) unmake(ply + d);

  tt_store_playout(node_key, value - node_spread);
  return value;
}

EndgameSolver::SearchResult EndgameSolver::search_child(int child_depth, int32_t alpha,
                                                        int32_t beta, int child_ply, bool first) {
  SearchResult r;
  if (first) {
    r = negamax(child_depth, -beta, -alpha, child_ply);
  } else {
    r = negamax(child_depth, -alpha - 1, -alpha, child_ply);
    if (alpha < -r.value && -r.value < beta) r = negamax(child_depth, -beta, -alpha, child_ply);
  }
  // Return the value from the parent's perspective; the proven bit is a property
  // of the search that produced the child's final verdict, not of the sign.
  return {-r.value, r.proven};
}

int32_t EndgameSolver::outplay_futility_bound(const Move& m, const OutplaySet& replier_outs) const {
  const int mover = stm_;
  // A move that ends the game -- an out-play, or a pass that trips the internal
  // scoreless cap -- gives the replier no turn, so no out-play bounds it.
  if (m.type() == MoveType::PLAY && m.num_glyphs() == racks_[mover].size()) return kInf;
  if (m.type() == MoveType::PASS && scoreless_ >= 1) return kInf;
  const int32_t p = best_surviving_score(replier_outs, m);
  if (p == kNoOutplaySurvivor) return kInf;
  // The replier can answer m with the surviving out-play: it scores p and banks
  // twice the face value L of the mover's post-m leftover as its end-of-game
  // bonus, so the game ends at spread s + m.score() - p - 2L for the mover,
  // where s is the mover's spread at this node. That line is available to the
  // replier, so it caps m's value; anything better for the mover would require
  // the replier to play worse.
  //
  // The bound stays sound when the out-play set is incomplete. Entries the set
  // is missing -- dropped by the halo kill filter without really being blocked,
  // or newly enabled by m -- are further replier options, which can only lower
  // m's true value. A conservatively dropped entry therefore costs a tighter
  // bound, never a valid one.
  const int32_t leftover = racks_[mover].point_value() - placed_face_value(m);
  return spread_stm() + m.score() - p - 2 * leftover;
}

EndgameSolver::SearchResult EndgameSolver::negamax(int depth, int32_t alpha, int32_t beta,
                                                   int ply) {
  if (aborting_) return {0, false};
  ++nodes_;
  if (nodes_ > budget_) {
    aborting_ = true;
    return {0, false};
  }
  if (game_over_) return {spread_stm(), true};  // a real game end is proven

  const uint64_t hash = node_hash();
  const int32_t alpha_orig = alpha;
  Move tt_move = Move::pass();
  bool have_tt_move = false;
  if (TTEntry* e = tt_probe(hash)) {
    // A depth-0 entry comes from a greedy playout and carries only a pass
    // placeholder in its move slot, so it supplies no ordering hint; its value
    // is still usable as an exact depth-0 result below.
    if (e->depth > 0) {
      have_tt_move = true;
      tt_move = e->best;
    }
    if (e->depth >= depth) {
      const int32_t score = e->score_rel + spread_stm();
      const bool proven = tt_is_proven(e->flag);
      const uint8_t bound = tt_bound(e->flag);
      if (bound == kExact) return {score, proven};
      if (bound == kLower) alpha = std::max(alpha, score);
      if (bound == kUpper) beta = std::min(beta, score);
      if (alpha >= beta) return {score, proven};
    }
  }

  if (depth == 0) return {greedy_playout(hash, ply), false};  // a playout leaf is an estimate

  // Owned copy: LeaveOutplays below holds a reference to this list, and the
  // child searches in the scan run nested generate_moves calls that may insert
  // into the memo, so an owned vector keeps the list stable across the loop.
  std::vector<Move> plays = generate_moves(racks_[stm_], ply);
  std::vector<RankedMove> moves;
  moves.reserve(plays.size() + 1);
  for (const Move& m : plays) moves.push_back({m, 0});
  moves.push_back({Move::pass(), 0});

  // The replier's maintained out-play set: every mover move that provably
  // leaves one of its entries intact is capped by its futility bound, first in
  // the ordering below and then as a pruning cutoff in the scan. LeaveOutplays
  // buckets this node's own play list to derive the children's mover-side sets.
  const OutplaySet* replier_outs = nullptr;
  if (outplay_futility_ && !outplay_sets_.current(1 - stm_).empty())
    replier_outs = &outplay_sets_.current(1 - stm_);
  std::optional<LeaveOutplays> leave_outs;
  if (outplay_futility_) leave_outs.emplace(board_, racks_[stm_], plays);
  order_moves(moves, tt_move, have_tt_move, replier_outs);

  int32_t best = -kInf;
  Move best_move = Move::pass();
  bool searched = false;    // at least one child was actually searched (for PVS)
  bool all_proven = true;   // AND over every scanned child's verdict
  bool cut = false;         // a beta cutoff fired
  bool cut_proven = false;  // proven bit of the child that witnessed the cutoff
  for (RankedMove& entry : moves) {
    const Move& move = entry.move;
    // Futility: a move a surviving replier out-play caps below alpha cannot
    // raise alpha. The bound comes from a terminal line, so skipping the move
    // neither changes the value nor clears the node's proven bit; fold the
    // bound in so a fail-low value stays sound even if every move is pruned.
    const int32_t ubound = replier_outs ? outplay_futility_bound(move, *replier_outs) : kInf;
    if (ubound <= alpha) {
      best = std::max(best, ubound);
      continue;
    }
    outplay_sets_.push(move, leave_outs ? &*leave_outs : nullptr, stm_, ply + 1);
    make(move, ply);
    const SearchResult child = search_child(depth - 1, alpha, beta, ply + 1, !searched);
    unmake(ply);
    outplay_sets_.pop(ply + 1);
    if (aborting_) return {best, false};  // an aborted scan proves nothing
    searched = true;
    all_proven = all_proven && child.proven;
    if (child.value > best) {
      best = child.value;
      best_move = move;
    }
    alpha = std::max(alpha, best);
    if (alpha >= beta) {
      cut = true;
      cut_proven = child.proven;  // the fail-high rests on this one witness
      break;
    }
  }

  // A fail-high is proven by its single witness; an exact or fail-low verdict
  // needs every child's contribution to be proven (pruned replies contribute a
  // proven terminal bound, so they leave all_proven untouched).
  const bool proven = cut ? cut_proven : all_proven;
  uint8_t bound = kExact;
  if (best <= alpha_orig)
    bound = kUpper;
  else if (best >= beta)
    bound = kLower;
  tt_store(hash, best - spread_stm(), bound, proven, best_move, depth);
  return {best, proven};
}

EndgameSolver::SearchResult EndgameSolver::run_root(int depth, int32_t alpha, int32_t beta,
                                                    bool first_win,
                                                    std::vector<RankedMove>& root_moves,
                                                    const std::vector<Move>& plays,
                                                    Move* best_out) {
  // The root is the solving side's node; it hands its children out-play sets
  // just as an interior node does. Under the narrow first-win window it also
  // futility-prunes its own moves like an interior node; under the full window
  // it never prunes, because every root move needs its exact value for the
  // re-ordering between iterations. It applies a beta cutoff as well (only
  // reachable under the narrow first-win window; see the scan below).
  std::optional<LeaveOutplays> leave_outs;
  if (outplay_futility_) leave_outs.emplace(board_, racks_[stm_], plays);
  const OutplaySet* replier_outs = nullptr;
  if (outplay_futility_ && root_futility_ && first_win && !outplay_sets_.current(1 - stm_).empty())
    replier_outs = &outplay_sets_.current(1 - stm_);

  int32_t best = -kInf;
  Move best_move = root_moves[0].move;
  bool searched = false;     // at least one root child was actually searched (for PVS)
  bool all_proven = true;    // AND over every root child's verdict
  bool best_proven = false;  // proven bit of the child that set `best`
  for (RankedMove& rm : root_moves) {
    // Root futility: a move a surviving replier out-play caps below alpha
    // cannot affect the class verdict. The bound comes from a terminal line,
    // so folding it into the fail-low value keeps the verdict proven, and it
    // becomes the move's rank so re-sorts sink it. Folded bounds never raise
    // alpha (the prune fires only at or below it) and never set best_move, so
    // a pruned move can be returned only when every root move is bounded below
    // alpha -- a proven loss, where any legal move is class-correct.
    const int32_t ubound = replier_outs ? outplay_futility_bound(rm.move, *replier_outs) : kInf;
    if (ubound <= alpha) {
      best = std::max(best, ubound);
      rm.rank = ubound;
      continue;
    }
    outplay_sets_.push(rm.move, leave_outs ? &*leave_outs : nullptr, stm_, 1);
    make(rm.move, 0);
    const SearchResult child = search_child(depth - 1, alpha, beta, 1, !searched);
    unmake(0);
    outplay_sets_.pop(1);
    if (aborting_) return {best, false};  // the aborted subtree's value is meaningless
    searched = true;
    rm.rank = child.value;
    all_proven = all_proven && child.proven;
    if (child.value > best) {
      best = child.value;
      best_move = rm.move;
      best_proven = child.proven;
    }
    alpha = std::max(alpha, best);
    // Root beta cutoff. Under the full window (spread pass / lexicographic
    // second pass) beta is +infinity, so this never fires and the whole root is
    // scanned as before. Under the narrow first-win window a root fail-high
    // (best >= beta means value >= kFirstWinBeta, the win verdict) settles
    // everything the window can express, so scanning further root moves adds
    // nothing to the class verdict. alpha rises only through best, so the child
    // that just pushed alpha to beta is the one that set best last -- best_move
    // and best_proven already hold its move and proven bit, the exact witness
    // the fail-high rests on. Root moves left unscanned keep their prior rank,
    // so the next iteration's re-sort orders them by stale values; that is
    // acceptable, since a proven cutoff ends the deepening and an unproven one
    // only reorders candidates.
    if (root_cutoff_ && alpha >= beta) break;
  }
  *best_out = best_move;
  // A root fail-high (best >= beta, reachable only under the narrow first-win
  // window) rests on the single best-achieving child -- the cutoff witness when
  // the scan broke early, or the last child to raise best otherwise -- so its
  // proven bit alone settles the verdict. Otherwise every child must be proven
  // (pruned root moves contribute proven terminal bounds, so they leave
  // all_proven untouched).
  const bool proven = best >= beta ? best_proven : all_proven;
  return {best, proven};
}

EndgameResult EndgameSolver::run_iterative(int32_t alpha, int32_t beta, bool first_win,
                                           std::vector<RankedMove>& root_moves,
                                           const std::vector<Move>& plays, int max_plies) {
  EndgameResult result;
  result.best = root_moves[0].move;
  for (int depth = 1; depth <= max_plies; ++depth) {
    Move best_move = root_moves[0].move;
    const SearchResult sr = run_root(depth, alpha, beta, first_win, root_moves, plays, &best_move);
    if (aborting_) {
      // Budget exhausted mid-iteration: the last completed iteration's result
      // stands. When not even the first iteration finished, fall back to the
      // best fully-searched root move of the partial pass -- root moves are
      // estimate-ordered, so the strongest candidates were searched first --
      // or, when no root move completed, the estimate-ordered first move.
      if (result.depth_completed == 0 && sr.value > -kInf) {
        result.best = best_move;
        result.value = sr.value;
      }
      break;
    }
    result.best = best_move;
    result.value = sr.value;
    result.depth_completed = depth;
    result.proven = sr.proven;
    if (trace_) {
      *trace_ << "  depth " << depth << ": best " << trace_move(best_move) << ", value " << sr.value
              << (sr.proven ? " (proven)" : " (estimate)") << ", nodes " << nodes_ << "\n";
    }
    // A proven iteration has resolved the search exactly: in the full window the
    // value is the true game spread, and in the first-win window a proven verdict
    // always settles the win/draw/loss class (a proven bound is itself true).
    // Deepening cannot change the answer, so stop.
    if (proof_early_exit_ && sr.proven && (!first_win || settles_first_win_class(sr.value))) break;
    // Re-order root moves by their returned values for the next depth.
    std::sort(root_moves.begin(), root_moves.end(), by_rank_desc);
  }
  result.nodes = nodes_;
  return result;
}

bool EndgameSolver::verify_move_class(const Move& m, int cls, const std::vector<Move>& plays,
                                      int max_plies) {
  std::optional<LeaveOutplays> leave_outs;
  if (outplay_futility_) leave_outs.emplace(board_, racks_[stm_], plays);
  outplay_sets_.push(m, leave_outs ? &*leave_outs : nullptr, stm_, 1);
  make(m, 0);
  bool verified = false;
  if (game_over_) {
    // m ended the game: the final spread's class is exact.
    verified = class_of(scores_[0] - scores_[1]) == cls;
  } else {
    // The child is searched from the opponent's perspective, so parent class
    // `cls` needs a proven child verdict of class -cls. Iterative deepening at
    // the narrow window, over the warm table, until the proof lands or the
    // shared budget runs out.
    for (int depth = 1; depth <= max_plies && !aborting_; ++depth) {
      const SearchResult sr = negamax(depth, kFirstWinAlpha, kFirstWinBeta, 1);
      if (aborting_) break;
      if (sr.proven && settles_first_win_class(sr.value)) {
        verified = -class_of(sr.value) == cls;
        break;
      }
    }
  }
  unmake(0);
  outplay_sets_.pop(1);
  return verified;
}

EndgameResult EndgameSolver::solve_class_first(std::vector<RankedMove>& root_moves,
                                               const std::vector<Move>& plays, int max_plies,
                                               bool refine_spread) {
  // First pass: prove the win/draw/loss class as cheaply as possible. The pass
  // is capped at half the budget: on a position whose class is not provable
  // within it, an uncapped pass would burn the entire cap and leave the spread
  // fallback below no budget at all -- the narrow-window move it would return
  // is chosen for a bound rather than for points, the worst of both objectives.
  // Class proofs are cheap when they land, so the cap loses few of them; the
  // reserved half is the insurance premium both objectives pay for a move that
  // means something when the proof does not arrive.
  const uint64_t full_budget = budget_;
  budget_ = full_budget / 2;
  if (trace_) *trace_ << "class pass (narrow window, budget " << budget_ << "):\n";
  const EndgameResult first =
    run_iterative(kFirstWinAlpha, kFirstWinBeta, /*first_win=*/true, root_moves, plays, max_plies);
  budget_ = full_budget;
  aborting_ = false;  // a class pass stopped by its half-cap frees the rest

  if (!first.proven || first.depth_completed < 1) {
    // No class proof in budget: margin-maximizing play is the class-robust
    // fallback (margin is slack against estimate error), over the warm table.
    if (trace_) *trace_ << "no class proof; spread fallback pass:\n";
    const EndgameResult spread =
      run_iterative(-kInf, kInf, /*first_win=*/false, root_moves, plays, max_plies);
    return spread.depth_completed >= 1 ? spread : first;
  }

  // The class is proven. Under the break-out objective that is the whole
  // answer -- the point of it is to stop spending on a decided endgame.
  if (!refine_spread) {
    EndgameResult result = first;
    result.proven_class = class_of(first.value);
    return result;
  }

  // The class is settled; the rest of the budget maximizes spread. Every move
  // of a proven-lost position is class-equal, so there the spread answer is
  // pure defense and needs no check. In a proven win or draw the spread answer
  // is played only if it provably preserves the class: a proven spread pass
  // implies that (the exact optimum's sign is the class), an unproven one must
  // pass a narrow-window probe of its chosen child, and on any failure the
  // proven-class move from the first pass stands.
  const int cls = class_of(first.value);
  if (trace_)
    *trace_ << "class proven: "
            << (cls > 0   ? "WIN"
                : cls < 0 ? "LOSS"
                          : "DRAW")
            << "; spread pass on the remaining budget:\n";
  EndgameResult result = first;
  result.proven_class = cls;
  const EndgameResult spread =
    run_iterative(-kInf, kInf, /*first_win=*/false, root_moves, plays, max_plies);
  if (spread.depth_completed < 1) return result;
  if (cls != -1 && !spread.proven && !verify_move_class(spread.best, cls, plays, max_plies))
    return result;
  result.best = spread.best;
  result.value = spread.value;
  result.depth_completed = spread.depth_completed;
  result.proven = spread.proven;
  return result;
}

void EndgameSolver::set_trace(std::ostream* os, MoveFormatter fmt) {
  trace_ = os;
  trace_fmt_ = fmt ? std::move(fmt) : [](const Board&, const Move&) { return std::string("?"); };
}

void EndgameSolver::trace_root_view(const std::vector<RankedMove>& root_moves) {
  std::ostream& os = *trace_;
  const OutplaySet& outs = outplay_sets_.root_replier();
  os << "replier out-plays on the current board (" << outs.size() << "):\n";
  for (const OutplayEntry& e : outs)
    os << "  " << trace_move(e.move) << " +" << e.move.score() << "\n";
  if (outs.empty()) {
    os << "  (none: no futility bounds apply at the root)\n";
    return;
  }

  // The block-or-outscore view of every root move m scoring g with leftover
  // face value L, against the strongest out-play (+p) m fails to block:
  //   final spread <= U(m) = s + g - p - 2L,
  // so m must either block every out-play or score at least p + 2L - s to
  // reach a draw. kInf bounds are annotated with why no bound applies.
  const int32_t s_root = spread_stm();
  int32_t best_bound = -kInf;
  bool any_unbounded = false;
  os << "root moves (spread " << (s_root >= 0 ? "+" : "") << s_root << "):\n";
  for (const RankedMove& rm : root_moves) {
    const Move& m = rm.move;
    os << "  " << trace_move(m);
    if (m.type() == MoveType::PLAY) os << " +" << m.score();
    const int32_t u = outplay_futility_bound(m, outs);
    if (u == kInf) {
      any_unbounded = true;
      const bool ends = (m.type() == MoveType::PLAY && m.num_glyphs() == racks_[stm_].size()) ||
                        (m.type() == MoveType::PASS && scoreless_ >= 1);
      os << (ends ? ": ends the game itself; no bound\n" : ": blocks every out-play; no bound\n");
      continue;
    }
    best_bound = std::max(best_bound, u);
    const Move* survivor = nullptr;
    for (const OutplayEntry& e : outs) {
      if (!move_touches_halo(e.halo, m)) {
        survivor = &e.move;
        break;
      }
    }
    const int32_t leftover = racks_[stm_].point_value() - placed_face_value(m);
    os << ": leaves " << trace_move(*survivor) << " (+" << best_surviving_score(outs, m)
       << ") alive; bound " << s_root << " + " << m.score() << " - "
       << best_surviving_score(outs, m) << " - 2*" << leftover << " = " << u << " (needs >= +"
       << best_surviving_score(outs, m) + 2 * leftover - s_root << " to reach a draw)\n";
  }
  if (!any_unbounded) {
    os << "no root move blocks every out-play or ends the game; best achievable bound "
       << best_bound << (best_bound < 0 ? " -> provably losing, pending search confirmation" : "")
       << "\n";
  }
}

bool EndgameSolver::reprove_walk_move(int ply, int req_class, Move* out) {
  for (int depth = 1; depth <= kMaxPlayout; ++depth) {
    const SearchResult sr = negamax(depth, kFirstWinAlpha, kFirstWinBeta, ply);
    if (!sr.proven || !settles_first_win_class(sr.value) || class_of(sr.value) != req_class)
      continue;
    // The proof's root entry (just stored, or the hit that answered it) holds
    // the class-preserving move; revalidate it against a fresh generation. The
    // generation runs even for a PASS (which needs no validation): it stamps
    // this walk ply's PathMoveLists slot, which deeper walk positions derive
    // their lists from -- a TT-answered proof would otherwise leave it stale.
    TTEntry* e = tt_probe(node_hash());
    if (e == nullptr) return false;
    const Move m = e->best;
    const std::vector<Move>& plays = generate_moves(racks_[stm_], ply);
    if (m.type() == MoveType::PLAY && std::find(plays.begin(), plays.end(), m) == plays.end()) {
      return false;
    }
    *out = m;
    return true;
  }
  return false;
}

void EndgameSolver::extract_continuation(EndgameResult& result) {
  if (trace_) *trace_ << "certificate walk: " << trace_move(result.best) << " (chosen move)\n";

  // Reconstruction runs after the search proper, outside the node budget and
  // with futility pruning off (the incremental out-play sets are not
  // maintained along the walk, so its re-searches must not consult them).
  const uint64_t nodes_before = nodes_;
  const uint64_t budget_before = budget_;
  const bool aborting_before = aborting_;
  const bool futility_before = outplay_futility_;
  budget_ = UINT64_MAX / 2;
  aborting_ = false;
  outplay_futility_ = false;

  make(result.best, 0);
  int ply = 1;
  bool ok = true;
  while (!game_over_ && ply < kMaxPlayout) {
    // The class from the current mover's perspective, under the walk invariant:
    // the winner's moves are re-proven below and the doomed side's moves cannot
    // change the class, so every position on the walk keeps the root's class.
    const int req_class = stm_ == 0 ? result.proven_class : -result.proven_class;
    Move next = Move::pass();
    if (req_class >= 0) {
      // This side must preserve a win or hold a draw: take a freshly-proven
      // class-preserving move.
      if (!reprove_walk_move(ply, req_class, &next)) {
        ok = false;
        break;
      }
    } else {
      // This side is proven lost: no move of theirs changes the class, so the
      // greedy playout move stands in for their optimal play.
      const std::vector<Move>& plays = generate_moves(racks_[stm_], ply);
      if (!plays.empty()) next = greedy_pick(plays);
    }
    if (trace_) *trace_ << "  " << trace_move(next) << "\n";
    result.continuation.push_back(next);
    make(next, ply++);
  }
  const int32_t final_spread = scores_[0] - scores_[1];
  ok = ok && game_over_ && class_of(final_spread) == result.proven_class;
  for (int d = ply - 1; d >= 0; --d) unmake(d);
  if (trace_) {
    *trace_ << "certificate " << (ok ? "valid" : "REJECTED") << ": final spread " << final_spread
            << " (class " << class_of(final_spread) << "), proven class " << result.proven_class
            << "\n";
  }
  if (!ok) result.continuation.clear();

  result.certificate_nodes = nodes_ - nodes_before;
  nodes_ = nodes_before;
  budget_ = budget_before;
  aborting_ = aborting_before;
  outplay_futility_ = futility_before;
}

EndgameResult EndgameSolver::solve(const EndgameState& state, const Params& params) {
  const Dictionary& dict = *state.dict;
  const uint64_t node_budget = params.budget;
  const int max_plies = params.plies;
  board_ = state.board;
  board_.ensure_movegen_caches(dict);
  dict_ = &dict;
  racks_[0] = state.my_rack;
  racks_[1] = state.opp_rack;
  scores_[0] = state.my_score;
  scores_[1] = state.opp_score;
  scoreless_ = state.scoreless_turns > 0 ? 1 : 0;
  stm_ = 0;
  game_over_ = false;
  board_hash_ = compute_board_hash();
  nodes_ = 0;
  movegens_ = 0;
  aborting_ = false;
  budget_ = node_budget;

  // Sized for the deepest search stack plus a certificate walk that runs its
  // re-searches (and their playouts) on top of the walked plies.
  const size_t needed = size_t(max_plies) + 3 * kMaxPlayout + 2;
  if (frames_.size() < needed) frames_.resize(needed);

  // Owned copy: `plays` is handed by const reference to run_iterative and on to
  // LeaveOutplays, so it must outlive every nested search (and any memo insert
  // those searches make) for the whole solve.
  std::vector<Move> plays = generate_moves_scratch(racks_[0]);
  std::vector<RankedMove> root_moves;
  root_moves.reserve(plays.size() + 1);
  for (const Move& m : plays) root_moves.push_back({m, 0});
  root_moves.push_back({Move::pass(), 0});
  order_moves(root_moves, Move::pass(), /*have_tt_move=*/false, /*replier_outs=*/nullptr);

  EndgameResult result;
  result.best = root_moves[0].move;
  // Searching a root move costs at least one node, so a position with more
  // root moves than the budget provably cannot complete its first iteration.
  // Decline it up front -- callers treat depth_completed == 0 as "unsolved"
  // and fall back to their own move policy -- rather than spending the whole
  // budget on a fraction of the root.
  if (root_moves.size() > node_budget) {
    result.movegens = movegens_;
    return result;
  }

  // Seed the incremental out-play sets from one move generation against the
  // opponent's rack. The same generation seeds PathMoveLists' root lists, from
  // which every deeper node's move list is derived. Seeded only once the solve
  // is sure to run, so a declined position pays nothing.
  outplay_sets_.reset(int(needed));
  if (outplay_futility_ || incremental_movegen_) {
    const std::vector<Move>& opp_plays = generate_moves_scratch(state.opp_rack);
    if (outplay_futility_)
      outplay_sets_.collect_root_replier(board_, opp_plays, state.opp_rack.size());
    if (incremental_movegen_) {
      path_lists_.reset(&board_, dict_, int(needed));
      path_lists_.set_root_list(0, plays);
      path_lists_.set_root_list(1, opp_plays);
    }
  }
  if (trace_) {
    *trace_ << "solve: spread-matters " << (params.spread_matters ? "true" : "false") << ", budget "
            << node_budget << ", scores " << state.my_score << "-" << state.opp_score << " (spread "
            << (state.my_score - state.opp_score >= 0 ? "+" : "")
            << state.my_score - state.opp_score << "), mover rack value "
            << state.my_rack.point_value() << ", replier rack value "
            << state.opp_rack.point_value() << "\n";
    trace_root_view(root_moves);
  }
  result = solve_class_first(root_moves, plays, max_plies, params.spread_matters);
  // A proven value's sign is the position's class; a result may already carry a
  // class proven by the first pass even when its final value is an unproven
  // spread refinement.
  if (result.proven && result.proven_class == EndgameResult::kClassUnknown)
    result.proven_class = class_of(result.value);
  if (result.proven_class != EndgameResult::kClassUnknown && result.depth_completed >= 1)
    extract_continuation(result);
  result.nodes = nodes_;
  result.movegens = movegens_;
  return result;
}

}  // namespace scribblez
