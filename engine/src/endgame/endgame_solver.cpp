#include "endgame/endgame_solver.h"

#include "agent/macondo_bot.h"
#include "endgame/outplays.h"
#include "game/glyph.h"
#include "game/movegen.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "util/math.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace scribblez {

namespace {

// A large finite bound for the alpha-beta window; real endgame spreads are far
// smaller, so it stands in for +/- infinity without overflow on negation.
constexpr int32_t kInf = 1'000'000;

// Upper bound on plies a single greedy leaf playout can add past the search
// depth. The combined racks hold at most 2*RACK_SIZE tiles, each play removes
// at least one, and two consecutive passes end the game, so a playout is always
// far shorter than this.
constexpr int kMaxPlayout = 40;

// Zobrist keys for every (square, letter, is-blank) placement, plus the keys
// mixed in at node level: one per internal scoreless count (0..2) and one for
// the solving side being to move. Deterministically derived from a fixed seed.
struct ZobristTable {
  std::array<std::array<std::array<uint64_t, 2>, 26>, BOARD_SIZE * BOARD_SIZE> square;
  std::array<uint64_t, 3> scoreless;
  uint64_t solving_to_move;
};

ZobristTable build_zobrist() {
  ZobristTable z;
  uint64_t s = 0xE3D1F0A2B4C6D8E9ULL;
  for (auto& sq : z.square)
    for (auto& letter : sq)
      for (auto& blank : letter) blank = util::splitmix64(++s);
  for (auto& v : z.scoreless) v = util::splitmix64(++s);
  z.solving_to_move = util::splitmix64(++s);
  return z;
}

// Descending order on a (move, estimated-value) pair's value: the comparator
// both the search and the root iterative-deepening loop sort candidate moves by.
bool by_value_desc(const std::pair<Move, int32_t>& a, const std::pair<Move, int32_t>& b) {
  return a.second > b.second;
}

// In the first_win window a proven result settles the win/draw/loss class iff
// its value is a proven win bound (>= kFirstWinBeta), a proven loss bound
// (<= kFirstWinAlpha), or a proven draw (0). A proven verdict in this window is
// always one of these, so this holds whenever the result is proven.
bool settles_first_win_class(int32_t value) {
  return value >= EndgameSolver::kFirstWinBeta || value <= EndgameSolver::kFirstWinAlpha ||
         value == 0;
}

const ZobristTable& zobrist() {
  static const ZobristTable table = build_zobrist();
  return table;
}

}  // namespace

EndgameSolver::EndgameSolver(int tt_log2_entries)
    : tt_(static_cast<size_t>(1) << tt_log2_entries),
      tt_mask_((static_cast<uint64_t>(1) << tt_log2_entries) - 1) {
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
  h ^= util::splitmix64(racks_[stm_].bits());
  if (stm_ == 0) h ^= z.solving_to_move;
  h ^= z.scoreless[scoreless_];
  return h;
}

int EndgameSolver::placed_face_value(const Move& move) const {
  int total = 0;
  for (int i = 0; i < move.num_glyphs(); ++i) total += move.glyph(i).value();
  return total;
}

void EndgameSolver::make(const Move& move, int ply) {
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
    est = static_cast<int32_t>(move.score()) - placed_face_value(move);
  } else {
    est = 0;  // PASS
  }
  if (have_tt_move && move == tt_move) est += (1 << 28);
  if (move.type() == MoveType::PASS && scoreless_ > 0) est += (1 << 29);
  return est;
}

void EndgameSolver::order_moves(std::vector<std::pair<Move, int32_t>>& moves, const Move& tt_move,
                                bool have_tt_move, const OutplaySet* replier_outs) const {
  for (auto& m : moves) {
    m.second = order_estimate(m.first, tt_move, have_tt_move);
    if (replier_outs == nullptr) continue;
    // A finite futility bound is a proven cap on the move's value. Rebasing it
    // to the current spread puts it in the same points-denominated units as the
    // estimates, and taking the min sinks provably weak moves -- including a
    // stale TT move whose payoff a replier out-play now caps.
    const int32_t u = outplay_futility_bound(m.first, *replier_outs);
    if (u < kInf) m.second = std::min(m.second, u - spread_stm());
  }
  std::sort(moves.begin(), moves.end(), by_value_desc);
}

double EndgameSolver::playout_adjusted(const Move& move) const {
  const int mover = stm_;
  const int placed = placed_face_value(move);
  const bool empties = move.num_glyphs() == racks_[mover].size();
  const int rack_pv = racks_[mover].point_value();
  const int term = empties ? 2 * racks_[1 - mover].point_value() : -2 * (rack_pv - placed) - 10;
  return static_cast<double>(move.score()) + term;
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
  e.depth = static_cast<uint8_t>(depth);
  e.gen = tt_gen_;
}

void EndgameSolver::tt_store_playout(uint64_t hash, int32_t score_rel) {
  TTEntry& e = tt_[hash & tt_mask_];
  // Never overwrite a current-generation real (deeper) search result with a
  // depth-0 playout value.
  if (e.gen == tt_gen_ && e.flag != kEmpty && e.depth > 0) return;
  e.hash = hash;
  e.best = Move::pass();
  e.score_rel = score_rel;
  e.flag = kExact;
  e.depth = 0;
  e.gen = tt_gen_;
}

int32_t EndgameSolver::greedy_playout(uint64_t node_key, int ply) {
  const int solving_seat = stm_;
  const int32_t node_spread = spread_stm();
  int made = 0;
  while (!game_over_ && made < kMaxPlayout) {
    const std::vector<Move> plays = MoveGenerator(board_, *dict_).generate(racks_[stm_]);
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
  // twice the face value of the mover's post-m leftover as its end-of-game
  // bonus, capping m's value at the resulting terminal spread. See the class
  // comment for the derivation and why missed out-plays keep the bound sound.
  const int32_t leftover = racks_[mover].point_value() - placed_face_value(m);
  return spread_stm() + m.score() - p - 2 * leftover;
}

void EndgameSolver::push_outplay_sets(const Move& m, LeaveOutplays* leave_outs, int child_ply,
                                      OutplaySet* saved[2]) {
  saved[0] = cur_sets_[0];
  saved[1] = cur_sets_[1];
  if (leave_outs == nullptr) return;  // futility disabled: the sets are never read
  PlyOutplaySets& slot = ply_sets_[child_ply];
  leave_outs->collect_after(m, slot.mover);
  assign_surviving(*cur_sets_[1 - stm_], m, slot.other);
  cur_sets_[stm_] = &slot.mover;
  cur_sets_[1 - stm_] = &slot.other;
}

void EndgameSolver::restore_outplay_sets(OutplaySet* const saved[2]) {
  cur_sets_[0] = saved[0];
  cur_sets_[1] = saved[1];
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
    have_tt_move = true;
    tt_move = e->best;
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

  std::vector<Move> plays = MoveGenerator(board_, *dict_).generate(racks_[stm_]);
  std::vector<std::pair<Move, int32_t>> moves;
  moves.reserve(plays.size() + 1);
  for (const Move& m : plays) moves.emplace_back(m, 0);
  moves.emplace_back(Move::pass(), 0);

  // The replier's maintained out-play set: every mover move that provably
  // leaves one of its entries intact is capped by its futility bound, first in
  // the ordering below and then as a pruning cutoff in the scan. LeaveOutplays
  // buckets this node's own play list to derive the children's mover-side sets.
  const OutplaySet* replier_outs = nullptr;
  if (outplay_futility_ && !cur_sets_[1 - stm_]->empty()) replier_outs = cur_sets_[1 - stm_];
  std::optional<LeaveOutplays> leave_outs;
  if (outplay_futility_) leave_outs.emplace(board_, racks_[stm_], plays);
  order_moves(moves, tt_move, have_tt_move, replier_outs);

  int32_t best = -kInf;
  Move best_move = Move::pass();
  bool searched = false;    // at least one child was actually searched (for PVS)
  bool all_proven = true;   // AND over every scanned child's verdict
  bool cut = false;         // a beta cutoff fired
  bool cut_proven = false;  // proven bit of the child that witnessed the cutoff
  for (auto& entry : moves) {
    const Move& move = entry.first;
    // Futility: a move a surviving replier out-play caps below alpha cannot
    // raise alpha. The bound comes from a terminal line, so skipping the move
    // neither changes the value nor clears the node's proven bit; fold the
    // bound in so a fail-low value stays sound even if every move is pruned.
    const int32_t ubound = replier_outs ? outplay_futility_bound(move, *replier_outs) : kInf;
    if (ubound <= alpha) {
      best = std::max(best, ubound);
      continue;
    }
    OutplaySet* saved_sets[2];
    push_outplay_sets(move, leave_outs ? &*leave_outs : nullptr, ply + 1, saved_sets);
    make(move, ply);
    const SearchResult child = search_child(depth - 1, alpha, beta, ply + 1, !searched);
    unmake(ply);
    restore_outplay_sets(saved_sets);
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

EndgameSolver::SearchResult EndgameSolver::run_root(
  int depth, int32_t alpha, int32_t beta, std::vector<std::pair<Move, int32_t>>& root_moves,
  const std::vector<Move>& plays, Move* best_out) {
  // The root is the solving side's node; it hands its children out-play sets
  // just as an interior node does, but is itself never futility-pruned: every
  // root move needs its exact value for the re-ordering between iterations.
  std::optional<LeaveOutplays> leave_outs;
  if (outplay_futility_) leave_outs.emplace(board_, racks_[stm_], plays);

  int32_t best = -kInf;
  Move best_move = root_moves[0].first;
  bool all_proven = true;    // AND over every root child's verdict
  bool best_proven = false;  // proven bit of the child that set `best`
  for (auto& rm : root_moves) {
    OutplaySet* saved_sets[2];
    push_outplay_sets(rm.first, leave_outs ? &*leave_outs : nullptr, 1, saved_sets);
    make(rm.first, 0);
    const SearchResult child = search_child(depth - 1, alpha, beta, 1, best == -kInf);
    unmake(0);
    restore_outplay_sets(saved_sets);
    if (aborting_) return {best, false};  // the aborted subtree's value is meaningless
    rm.second = child.value;
    all_proven = all_proven && child.proven;
    if (child.value > best) {
      best = child.value;
      best_move = rm.first;
      best_proven = child.proven;
    }
    alpha = std::max(alpha, best);
  }
  *best_out = best_move;
  // The root never applies a beta cutoff, but the returned value can still be a
  // fail-high bound when alpha rose to beta (a narrow first_win window): then the
  // "value >= beta" verdict rests on the single best-achieving child, so its
  // proven bit alone settles it. Otherwise every child must be proven.
  const bool proven = best >= beta ? best_proven : all_proven;
  return {best, proven};
}

EndgameResult EndgameSolver::solve(const Board& board, const Dictionary& dict, const Rack& my_rack,
                                   const Rack& opp_rack, int my_score, int opp_score,
                                   int scoreless_turns, uint64_t node_budget, int max_plies,
                                   bool first_win) {
  board_ = board;
  board_.ensure_movegen_caches(dict);
  dict_ = &dict;
  racks_[0] = my_rack;
  racks_[1] = opp_rack;
  scores_[0] = my_score;
  scores_[1] = opp_score;
  scoreless_ = scoreless_turns > 0 ? 1 : 0;
  stm_ = 0;
  game_over_ = false;
  board_hash_ = compute_board_hash();
  nodes_ = 0;
  aborting_ = false;
  budget_ = node_budget;

  const size_t needed = static_cast<size_t>(max_plies) + kMaxPlayout + 2;
  if (frames_.size() < needed) frames_.resize(needed);
  const size_t sets_needed = static_cast<size_t>(max_plies) + 2;
  if (ply_sets_.size() < sets_needed) ply_sets_.resize(sets_needed);

  std::vector<Move> plays = MoveGenerator(board_, dict).generate(my_rack);
  std::vector<std::pair<Move, int32_t>> root_moves;
  root_moves.reserve(plays.size() + 1);
  for (const Move& m : plays) root_moves.emplace_back(m, 0);
  root_moves.emplace_back(Move::pass(), 0);
  order_moves(root_moves, Move::pass(), /*have_tt_move=*/false, /*replier_outs=*/nullptr);

  EndgameResult result;
  result.best = root_moves[0].first;
  // Searching a root move costs at least one node, so a position with more
  // root moves than the budget provably cannot complete its first iteration.
  // Decline it up front -- callers treat depth_completed == 0 as "unsolved"
  // and fall back to their own move policy -- rather than spending the whole
  // budget on a fraction of the root.
  if (root_moves.size() > node_budget) return result;

  // Seed the incremental out-play sets (see the class comment): the opponent's
  // from one move generation against its rack, the solving side's empty -- the
  // root node itself is never futility-pruned, and its children's mover-side
  // sets come from bucketing the root play list. Seeded only once the solve is
  // sure to run, so a declined position pays nothing.
  root_sets_[0].clear();
  root_sets_[1].clear();
  cur_sets_[0] = &root_sets_[0];
  cur_sets_[1] = &root_sets_[1];
  if (outplay_futility_) {
    collect_rack_outplays(board_, MoveGenerator(board_, dict).generate(opp_rack), opp_rack.size(),
                          root_sets_[1]);
  }
  const int32_t root_alpha = first_win ? kFirstWinAlpha : -kInf;
  const int32_t root_beta = first_win ? kFirstWinBeta : kInf;
  for (int depth = 1; depth <= max_plies; ++depth) {
    Move best_move = root_moves[0].first;
    const SearchResult sr = run_root(depth, root_alpha, root_beta, root_moves, plays, &best_move);
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
    // A proven iteration has resolved the search exactly: in the full window the
    // value is the true game spread, and in the first_win window a proven verdict
    // always settles the win/draw/loss class (a proven bound is itself true).
    // Deepening cannot change the answer, so stop.
    if (proof_early_exit_ && sr.proven && (!first_win || settles_first_win_class(sr.value))) break;
    // Re-order root moves by their exact returned values for the next depth.
    std::sort(root_moves.begin(), root_moves.end(), by_value_desc);
  }
  result.nodes = nodes_;
  return result;
}

}  // namespace scribblez
