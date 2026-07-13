#include "endgame/endgame_solver.h"

#include "agent/macondo_bot.h"
#include "game/glyph.h"
#include "game/movegen.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "util/math.h"

#include <algorithm>
#include <array>
#include <cstdint>

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
                                bool have_tt_move) const {
  for (auto& m : moves) m.second = order_estimate(m.first, tt_move, have_tt_move);
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

void EndgameSolver::tt_store(uint64_t hash, int32_t score_rel, uint8_t flag, const Move& best,
                             int depth) {
  TTEntry& e = tt_[hash & tt_mask_];
  e.hash = hash;
  e.best = best;
  e.score_rel = score_rel;
  e.flag = flag;
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

int32_t EndgameSolver::negamax(int depth, int32_t alpha, int32_t beta, int ply) {
  if (aborting_) return 0;
  ++nodes_;
  if (nodes_ > budget_) {
    aborting_ = true;
    return 0;
  }
  if (game_over_) return spread_stm();

  const uint64_t hash = node_hash();
  const int32_t alpha_orig = alpha;
  Move tt_move = Move::pass();
  bool have_tt_move = false;
  if (TTEntry* e = tt_probe(hash)) {
    have_tt_move = true;
    tt_move = e->best;
    if (e->depth >= depth) {
      const int32_t score = e->score_rel + spread_stm();
      if (e->flag == kExact) return score;
      if (e->flag == kLower) alpha = std::max(alpha, score);
      if (e->flag == kUpper) beta = std::min(beta, score);
      if (alpha >= beta) return score;
    }
  }

  if (depth == 0) return greedy_playout(hash, ply);

  std::vector<Move> plays = MoveGenerator(board_, *dict_).generate(racks_[stm_]);
  std::vector<std::pair<Move, int32_t>> moves;
  moves.reserve(plays.size() + 1);
  for (const Move& m : plays) moves.emplace_back(m, 0);
  moves.emplace_back(Move::pass(), 0);
  order_moves(moves, tt_move, have_tt_move);

  int32_t best = -kInf;
  Move best_move = Move::pass();
  for (size_t i = 0; i < moves.size(); ++i) {
    make(moves[i].first, ply);
    int32_t value;
    if (i == 0) {
      value = -negamax(depth - 1, -beta, -alpha, ply + 1);
    } else {
      value = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
      if (alpha < value && value < beta) value = -negamax(depth - 1, -beta, -alpha, ply + 1);
    }
    unmake(ply);
    if (aborting_) return best;
    if (value > best) {
      best = value;
      best_move = moves[i].first;
    }
    alpha = std::max(alpha, best);
    if (alpha >= beta) break;
  }

  uint8_t flag = kExact;
  if (best <= alpha_orig)
    flag = kUpper;
  else if (best >= beta)
    flag = kLower;
  tt_store(hash, best - spread_stm(), flag, best_move, depth);
  return best;
}

int32_t EndgameSolver::run_root(int depth, int32_t alpha, int32_t beta,
                                std::vector<std::pair<Move, int32_t>>& root_moves, Move* best_out) {
  int32_t best = -kInf;
  Move best_move = root_moves[0].first;
  for (auto& rm : root_moves) {
    make(rm.first, 0);
    int32_t value;
    if (best == -kInf) {
      value = -negamax(depth - 1, -beta, -alpha, 1);
    } else {
      value = -negamax(depth - 1, -alpha - 1, -alpha, 1);
      if (alpha < value && value < beta) value = -negamax(depth - 1, -beta, -alpha, 1);
    }
    unmake(0);
    if (aborting_) break;  // the aborted subtree's value is meaningless; keep best-so-far
    rm.second = value;
    if (value > best) {
      best = value;
      best_move = rm.first;
    }
    alpha = std::max(alpha, best);
  }
  *best_out = best_move;
  return best;
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

  std::vector<Move> plays = MoveGenerator(board_, dict).generate(my_rack);
  std::vector<std::pair<Move, int32_t>> root_moves;
  root_moves.reserve(plays.size() + 1);
  for (const Move& m : plays) root_moves.emplace_back(m, 0);
  root_moves.emplace_back(Move::pass(), 0);
  order_moves(root_moves, Move::pass(), /*have_tt_move=*/false);

  EndgameResult result;
  result.best = root_moves[0].first;
  // Searching a root move costs at least one node, so a position with more
  // root moves than the budget provably cannot complete its first iteration.
  // Decline it up front -- callers treat depth_completed == 0 as "unsolved"
  // and fall back to their own move policy -- rather than spending the whole
  // budget on a fraction of the root.
  if (root_moves.size() > node_budget) return result;
  const int32_t root_alpha = first_win ? kFirstWinAlpha : -kInf;
  const int32_t root_beta = first_win ? kFirstWinBeta : kInf;
  for (int depth = 1; depth <= max_plies; ++depth) {
    Move best_move = root_moves[0].first;
    const int32_t value = run_root(depth, root_alpha, root_beta, root_moves, &best_move);
    if (aborting_) {
      // Budget exhausted mid-iteration: the last completed iteration's result
      // stands. When not even the first iteration finished, fall back to the
      // best fully-searched root move of the partial pass -- root moves are
      // estimate-ordered, so the strongest candidates were searched first --
      // or, when no root move completed, the estimate-ordered first move.
      if (result.depth_completed == 0 && value > -kInf) {
        result.best = best_move;
        result.value = value;
      }
      break;
    }
    result.best = best_move;
    result.value = value;
    result.depth_completed = depth;
    // Re-order root moves by their exact returned values for the next depth.
    std::sort(root_moves.begin(), root_moves.end(), by_value_desc);
  }
  result.nodes = nodes_;
  return result;
}

}  // namespace scribblez
