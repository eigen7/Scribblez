#include "endgame/pre_endgame_solver.h"

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "util/assert.h"
#include "util/exception.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

namespace scribblez {

namespace {

// In the pre-endgame, as in the endgame, two consecutive scoreless turns end
// the game (Macondo's endgame mode).
constexpr int kMaxScoreless = 2;
// The ordering bonus of a pass that answers a pass, which ends the game.
constexpr int kEarlyPassBonus = 1 << 13;
// A play that leaves more tiles than this in the bag is not searched.
constexpr int kMaxTilesLeft = 1;
// How deep "our turn again with tiles in the bag" may nest.
constexpr int kNestedDepthLimit = 1;

// ---- Tallies ---------------------------------------------------------------

// Worse outcomes compare greater, so a max is the pessimistic combination.
enum class Outcome : uint8_t { kUnset, kWin, kDraw, kLoss };

// An ordered bag content: tiles[0] is drawn first.
using Draw = std::vector<Tile>;

struct Perm {
  Draw tiles;
  int count = 0;  // the ways the unseen tiles can yield this ordered draw
};

void permutations_rec(TileCounts& left, const TileCounts& orig, int k, Draw& cur,
                      std::vector<Perm>& out) {
  if (k == 0) {
    int count = 1;
    TileCounts avail = orig;
    for (Tile t : cur) {
      count *= avail.count(t);
      avail.remove(t);
    }
    out.push_back({cur, count});
    return;
  }
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    if (left.count(t) == 0) continue;
    left.remove(t);
    cur.push_back(t);
    permutations_rec(left, orig, k - 1, cur, out);
    cur.pop_back();
    left.add(t);
  }
}

// Every distinct ordered draw of `k` tiles from `pool`, with its multiplicity.
std::vector<Perm> permutations(const TileCounts& pool, int k) {
  std::vector<Perm> out;
  TileCounts left = pool;
  Draw cur;
  permutations_rec(left, pool, k, cur, out);
  return out;
}

struct DrawOutcome {
  Draw tiles;
  int count = 0;
  Outcome outcome = Outcome::kUnset;
  bool finalized = false;
};

// One play's results per bag draw (Macondo's PreEndgamePlay). An outcome is
// final once no later line can change it; until then it only worsens, since
// the opponent picks the reply that is worst for us.
struct PegPlay {
  Move move;
  double points = 0.0;
  double found_losses = 0.0;
  int64_t spread = 0;
  bool spread_set = false;
  std::vector<DrawOutcome> outcomes;

  const DrawOutcome* find(const Draw& tiles) const;
  bool has_loss(const Draw& tiles) const;
  void add_final(Outcome o, int count, const Draw& tiles);
  void set_unfinalized(Outcome o, int count, const Draw& tiles);
  void finalize();
  void add_spread(int32_t s, int count);
  int total() const;

 private:
  DrawOutcome& entry(const Draw& tiles);
  void credit(Outcome o, int count);
};

const DrawOutcome* PegPlay::find(const Draw& tiles) const {
  for (const DrawOutcome& d : outcomes)
    if (d.tiles == tiles) return &d;
  return nullptr;
}

bool PegPlay::has_loss(const Draw& tiles) const {
  const DrawOutcome* d = find(tiles);
  return d != nullptr && d->outcome == Outcome::kLoss;
}

DrawOutcome& PegPlay::entry(const Draw& tiles) {
  for (DrawOutcome& d : outcomes)
    if (d.tiles == tiles) return d;
  outcomes.push_back({.tiles = tiles});
  return outcomes.back();
}

void PegPlay::credit(Outcome o, int count) {
  if (o == Outcome::kWin) {
    points += count;
  } else if (o == Outcome::kDraw) {
    points += count / 2.0;
    found_losses += count / 2.0;
  } else if (o == Outcome::kLoss) {
    found_losses += count;
  }
}

void PegPlay::add_final(Outcome o, int count, const Draw& tiles) {
  DrawOutcome& d = entry(tiles);
  d.outcome = o;
  d.count += count;
  credit(o, count);
  d.finalized = true;
}

void PegPlay::set_unfinalized(Outcome o, int count, const Draw& tiles) {
  DrawOutcome& d = entry(tiles);
  if (d.outcome == Outcome::kUnset) d.count = count;
  d.outcome = std::max(d.outcome, o);
}

void PegPlay::finalize() {
  for (DrawOutcome& d : outcomes) {
    if (d.finalized) continue;
    credit(d.outcome, d.count);
    d.finalized = true;
  }
}

void PegPlay::add_spread(int32_t s, int count) {
  spread_set = true;
  spread += int64_t(s) * count;
}

int PegPlay::total() const {
  int n = 0;
  for (const DrawOutcome& d : outcomes) n += d.count;
  return n;
}

// ---- The game --------------------------------------------------------------

// The pre-endgame's game. Seat 0 is the mover being solved for.
struct PegGame {
  Board board;
  std::array<Rack, 2> racks;
  Draw bag;  // bag[0] is drawn first
  std::array<int, 2> scores{0, 0};
  int scoreless = 0;
  bool last_scoreless = false;  // the last move scored nothing
  int on_turn = 0;
  bool over = false;

  // Plays `m` for the side to move and draws its replacements.
  void play(const Move& m);
  // Deals the draw: the bag becomes `tiles`, and the opponent of seat 0 holds
  // the rest of `unseen` (seat 0's unseen tiles).
  void stage(const TileCounts& unseen, const Draw& tiles);
};

void PegGame::play(const Move& m) {
  Rack& rack = racks[on_turn];
  for (int i = 0; i < m.num_glyphs(); ++i) {
    const bool ok = rack.remove(m.glyph(i).rack_tile());
    RELEASE_ASSERT(ok);
  }
  last_scoreless = m.type() != MoveType::PLAY;
  if (m.type() == MoveType::PLAY) {
    board.apply(m);
    scores[on_turn] += m.score();
    scoreless = 0;
    const int draw = std::min<int>(m.num_glyphs(), bag.size());
    for (int i = 0; i < draw; ++i) rack.add(bag[i]);
    bag.erase(bag.begin(), bag.begin() + draw);
    if (rack.empty()) {
      scores[on_turn] += 2 * racks[1 - on_turn].point_value();
      over = true;
      return;
    }
  } else if (++scoreless >= kMaxScoreless) {
    for (int p = 0; p < 2; ++p) scores[p] -= racks[p].point_value();
    over = true;
    return;
  }
  on_turn = 1 - on_turn;
}

void PegGame::stage(const TileCounts& unseen, const Draw& tiles) {
  TileCounts opp = unseen;
  for (Tile t : tiles) opp.remove(t);
  racks[1] = Rack::from_counts(opp);
  bag = tiles;
}

MoveRequest request(const PegGame& g, const Dictionary& dict) {
  const int on = g.on_turn;
  return MoveRequest{
    g.board, dict, g.racks[on], g.racks[1 - on], g.scores[on], g.scores[1 - on], int(g.bag.size())};
}

// The static equity of the best move for the side to move, a pass included:
// Macondo's estimate of how hard a draw is, which orders the search so losses
// surface early.
double best_equity(const PegGame& g, const Dictionary& dict) {
  const MoveRequest req = request(g, dict);
  const HastyEquity& eq = HastyEquity::instance();
  const int on = g.on_turn;
  const double pass = eq.equity(Move::pass(), g.board, req.bag_size, g.racks[1 - on], g.racks[on]);
  const Move best = hasty_best_move_wmp(req);
  return std::max(pass, eq.equity(best, g.board, req.bag_size, g.racks[1 - on], g.racks[on]));
}

int sign(int64_t v) { return (v > 0) - (v < 0); }

Outcome outcome_of_class(int cls) {
  return cls > 0 ? Outcome::kWin : cls < 0 ? Outcome::kLoss : Outcome::kDraw;
}

template <class T, class Key>
void stable_sort_desc(std::vector<T>& v, Key key) {
  std::stable_sort(v.begin(), v.end(), [&](const T& a, const T& b) { return key(a) > key(b); });
}

// ---- The search ------------------------------------------------------------

// One iterative-deepening pass: every play scored at one oracle effort.
class PegSearch {
 public:
  PegSearch(const Dictionary& dict, EndgameOracle& oracle, UnprovenPolicy unproven, int num_in_bag,
            const TileCounts& unseen, int effort)
      : dict_(dict),
        oracle_(oracle),
        unproven_(unproven),
        num_in_bag_(num_in_bag),
        num_combos_(num_combos(num_in_bag)),
        unseen_(unseen),
        effort_(effort) {}

  // Scores `peg` over every draw of `maybe_in_bag` (Macondo's
  // processJobPerPlay), unless the early cutoff proves it cannot win.
  void process_play(PegPlay& peg, const PegGame& root, const TileCounts& maybe_in_bag);

  // Among the plays tied for first, solves the bag-emptying ones' endgames for
  // spread and ranks by it (Macondo's maybeTiebreak). `plays` is sorted by
  // points and stays sorted, the tied plays reordered by spread.
  void tiebreak(std::vector<PegPlay>& plays, const PegGame& root, const TileCounts& maybe_in_bag);

 private:
  struct Option {
    Draw tiles;
    int count = 0;
  };

  // The ordered draws of `n` tiles out of 7 + n unseen: the denominator of the
  // early cutoff.
  static int num_combos(int n);

  void recursive_solve(PegPlay& peg, const Move& to_make, const Option& opt, const PegGame& before,
                       bool empties_bag, bool full_solve, int nested_depth, bool outer);
  void solve_leaf(PegPlay& peg, const Option& opt, const PegGame& g, bool empties_bag,
                  bool full_solve, bool outer);
  void iterate_opp_replies(PegPlay& peg, const Option& opt, const PegGame& g, const Move& prev,
                           bool empties_bag, bool full_solve, int nested_depth, bool outer);
  void iterate_our_replies(PegPlay& peg, const Option& opt, const PegGame& g, bool empties_bag,
                           int nested_depth);
  Outcome nested_our_turn(const PegGame& g, int nested_depth);

  // Every move for the side to move, the pass included when gen_pass_ is set
  // or nothing else is legal.
  std::vector<Move> moves_for(const PegGame& g) const;
  std::vector<Move> sorted_replies(const PegGame& g, const Move& prev) const;
  std::vector<Move> sorted_our_moves(const PegGame& g) const;
  std::vector<Perm> sorted_sub_perms(const PegGame& g, const TileCounts& unseen, int k) const;
  // The class of a bag-empty leaf for seat 0.
  int leaf_class(const EndgameVerdict& v, bool we_move) const;
  std::string info_state_key(const PegGame& g) const;
  void note_winner(const PegPlay& peg);

  const Dictionary& dict_;
  EndgameOracle& oracle_;
  UnprovenPolicy unproven_;
  int num_in_bag_;
  int num_combos_;
  TileCounts unseen_;  // seat 0's unseen tiles at the root
  int effort_;
  bool gen_pass_ = true;
  double min_potential_losses_ = 100000.0;
  // Nested verdicts per information state, per actual bag draw.
  std::map<std::string, std::map<Draw, Outcome>> nested_cache_;
};

int PegSearch::num_combos(int n) {
  int c = 1;
  for (int i = 0; i < n; ++i) c *= RACK_SIZE + n - i;
  return c;
}

std::vector<Move> PegSearch::moves_for(const PegGame& g) const {
  std::vector<Move> moves = generate_legal_plays(request(g, dict_));
  if (gen_pass_ || moves.empty()) moves.push_back(Move::pass());
  return moves;
}

std::vector<Move> PegSearch::sorted_replies(const PegGame& g, const Move& prev) const {
  std::vector<Move> moves = moves_for(g);
  const bool after_pass = prev.type() == MoveType::PASS;
  stable_sort_desc(moves, [&](const Move& m) {
    return int(m.score()) + (after_pass && m.type() == MoveType::PASS ? kEarlyPassBonus : 0);
  });
  return moves;
}

std::vector<Move> PegSearch::sorted_our_moves(const PegGame& g) const {
  std::vector<Move> moves = moves_for(g);
  const HastyEquity& eq = HastyEquity::instance();
  const Rack& rack = g.racks[g.on_turn];
  // Score plus leave value, truncated to an integer as Macondo stores it.
  stable_sort_desc(moves, [&](const Move& m) {
    Rack leave = rack;
    for (int i = 0; i < m.num_glyphs(); ++i) leave.remove(m.glyph(i).rack_tile());
    return int(double(m.score()) + eq.leave_value(leave));
  });
  return moves;
}

std::vector<Perm> PegSearch::sorted_sub_perms(const PegGame& g, const TileCounts& unseen,
                                              int k) const {
  std::vector<Perm> perms = permutations(unseen, k);
  std::vector<std::pair<double, Perm>> est;
  for (const Perm& p : perms) {
    PegGame s = g;
    s.stage(unseen, p.tiles);
    s.on_turn = 1;
    est.push_back({best_equity(s, dict_), p});
  }
  stable_sort_desc(est, [](const auto& e) { return e.first; });
  for (size_t i = 0; i < perms.size(); ++i) perms[i] = est[i].second;
  return perms;
}

int PegSearch::leaf_class(const EndgameVerdict& v, bool we_move) const {
  int cls = v.cls;
  if (cls == EndgameVerdict::kClassUnknown) {
    if (unproven_ == UnprovenPolicy::kLoss) return -1;
    cls = sign(v.spread);
  }
  return we_move ? cls : -cls;
}

void PegSearch::note_winner(const PegPlay& peg) {
  min_potential_losses_ = std::min(min_potential_losses_, num_combos_ - peg.points);
}

void PegSearch::solve_leaf(PegPlay& peg, const Option& opt, const PegGame& g, bool empties_bag,
                           bool full_solve, bool outer) {
  int cls;
  if (g.over) {
    cls = sign(g.scores[0] - g.scores[1]);
  } else {
    const int on = g.on_turn;
    const EndgameState state{&dict_,       g.board,          g.racks[on], g.racks[1 - on],
                             g.scores[on], g.scores[1 - on], g.scoreless};
    const EndgameVerdict v =
      oracle_.evaluate(state, effort_, full_solve ? EndgameGoal::kSpread : EndgameGoal::kClass);
    if (full_solve && outer) {
      peg.add_spread(on == 0 ? v.spread : -v.spread, opt.count);
      return;
    }
    cls = leaf_class(v, on == 0);
  }
  const Outcome o = outcome_of_class(cls);
  if (empties_bag) {
    peg.add_final(o, opt.count, opt.tiles);
  } else {
    peg.set_unfinalized(o, opt.count, opt.tiles);
  }
  // A play that leaves tiles in the bag has unsettled points until finalized.
  if (empties_bag && outer) note_winner(peg);
}

void PegSearch::recursive_solve(PegPlay& peg, const Move& to_make, const Option& opt,
                                const PegGame& before, bool empties_bag, bool full_solve,
                                int nested_depth, bool outer) {
  if (!full_solve && peg.has_loss(opt.tiles)) return;
  if (before.over || before.bag.empty()) {
    solve_leaf(peg, opt, before, empties_bag, full_solve, outer);
    return;
  }
  PegGame g = before;
  g.play(to_make);
  if (g.over || g.bag.empty()) {
    solve_leaf(peg, opt, g, empties_bag, full_solve, outer);
  } else if (g.on_turn == 0) {
    iterate_our_replies(peg, opt, g, empties_bag, nested_depth);
  } else {
    iterate_opp_replies(peg, opt, g, to_make, empties_bag, full_solve, nested_depth, outer);
  }
}

// Pessimistic: the draw is lost as soon as any reply loses it.
void PegSearch::iterate_opp_replies(PegPlay& peg, const Option& opt, const PegGame& g,
                                    const Move& prev, bool empties_bag, bool full_solve,
                                    int nested_depth, bool outer) {
  for (const Move& reply : sorted_replies(g, prev)) {
    recursive_solve(peg, reply, opt, g, empties_bag, full_solve, nested_depth, outer);
    if (!full_solve && peg.has_loss(opt.tiles)) break;
  }
}

// Our turn with tiles still in the bag: we cannot see the draw, so a nested
// solve picks our reply over every draw consistent with what we know. Past the
// depth limit the draw is left unscored.
void PegSearch::iterate_our_replies(PegPlay& peg, const Option& opt, const PegGame& g,
                                    bool empties_bag, int nested_depth) {
  if (nested_depth + 1 > kNestedDepthLimit) return;
  const Outcome o = nested_our_turn(g, nested_depth + 1);
  if (empties_bag) {
    peg.add_final(o, opt.count, opt.tiles);
  } else {
    peg.set_unfinalized(o, opt.count, opt.tiles);
  }
}

std::string PegSearch::info_state_key(const PegGame& g) const {
  std::string key;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) key.push_back(char(g.board.at(r, c).code()));
  TileCounts unseen = g.racks[1].counts();
  for (Tile t : g.bag) unseen.add(t);
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    key.push_back(char(g.racks[0].count(t)));
    key.push_back(char(unseen.count(t)));
  }
  key.push_back(char(g.scoreless));
  return key;
}

// Scores each of our moves over every draw of the unseen tiles (bag plus the
// opponent's rack), keeps the moves with the fewest losses, and returns, for
// the actual draw, the worst outcome among them: we would not know which of the
// tied moves we would pick. Memoized per information state.
Outcome PegSearch::nested_our_turn(const PegGame& g, int nested_depth) {
  const std::string key = info_state_key(g);
  if (const auto it = nested_cache_.find(key); it != nested_cache_.end()) {
    if (const auto v = it->second.find(g.bag); v != it->second.end()) return v->second;
  }
  TileCounts unseen = g.racks[1].counts();
  for (Tile t : g.bag) unseen.add(t);
  const int sub_bag = g.bag.size();
  const std::vector<Perm> sub_perms = sorted_sub_perms(g, unseen, sub_bag);

  // Macondo generates our pass only when answering a pass, and the flag holds
  // for the replies searched below it.
  const bool saved_gen_pass = gen_pass_;
  gen_pass_ = g.last_scoreless;
  const std::vector<Move> ours = sorted_our_moves(g);

  std::map<Draw, Outcome> verdicts;
  std::vector<PegPlay> subs(ours.size());
  double min_losses = DBL_MAX;
  bool early_win = false;
  for (size_t i = 0; i < ours.size() && !early_win; ++i) {
    PegPlay& sub = subs[i];
    sub.move = ours[i];
    const bool empties = ours[i].num_glyphs() >= sub_bag;
    for (const Perm& sp : sub_perms) {
      PegGame s = g;
      s.stage(unseen, sp.tiles);
      recursive_solve(sub, ours[i], {sp.tiles, sp.count}, s, empties, false, nested_depth, false);
      // Its losses already exceed the leader's: it can never tie.
      if (sub.found_losses > min_losses) break;
    }
    sub.finalize();
    if (sub.found_losses < min_losses) {
      min_losses = sub.found_losses;
      // Wins every draw, all of them evaluated: nothing can do better.
      if (min_losses == 0 && sub.outcomes.size() == sub_perms.size()) {
        for (const DrawOutcome& d : sub.outcomes) verdicts[d.tiles] = Outcome::kWin;
        early_win = true;
      }
    }
  }
  gen_pass_ = saved_gen_pass;

  if (!early_win) {
    std::vector<const PegPlay*> tied;
    for (const PegPlay& p : subs)
      if (p.found_losses <= min_losses + 1e-4) tied.push_back(&p);
    if (tied.empty()) {
      for (const Perm& sp : sub_perms) verdicts[sp.tiles] = Outcome::kLoss;
    } else {
      for (const DrawOutcome& d : tied[0]->outcomes) {
        Outcome worst = d.outcome;
        for (const PegPlay* p : tied)
          if (const DrawOutcome* o = p->find(d.tiles)) worst = std::max(worst, o->outcome);
        verdicts[d.tiles] = worst;
      }
    }
  }
  const auto v = verdicts.find(g.bag);
  const Outcome verdict = v == verdicts.end() ? Outcome::kUnset : v->second;
  nested_cache_[key] = std::move(verdicts);
  return verdict;
}

void PegSearch::process_play(PegPlay& peg, const PegGame& root, const TileCounts& maybe_in_bag) {
  if (num_in_bag_ - peg.move.num_glyphs() > kMaxTilesLeft) return;
  const bool empties = peg.move.num_glyphs() >= num_in_bag_;
  // The draws hardest for us first: those where the opponent's best reply to
  // our play is worth the most. Losses then surface early.
  std::vector<std::pair<double, Perm>> options;
  for (const Perm& p : permutations(maybe_in_bag, num_in_bag_)) {
    PegGame g = root;
    g.stage(unseen_, p.tiles);
    g.play(peg.move);
    options.push_back({g.over ? 0.0 : best_equity(g, dict_), p});
  }
  stable_sort_desc(options, [](const auto& o) { return o.first; });
  for (const auto& [est, p] : options) {
    // It has already lost more than the leader can: it cannot win.
    if (peg.found_losses > min_potential_losses_) return;
    PegGame g = root;
    g.stage(unseen_, p.tiles);
    recursive_solve(peg, peg.move, {p.tiles, p.count}, g, empties, false, 0, true);
    peg.finalize();
  }
}

void PegSearch::tiebreak(std::vector<PegPlay>& plays, const PegGame& root,
                         const TileCounts& maybe_in_bag) {
  size_t tied = 1;
  while (tied < plays.size() && plays[tied].points == plays[0].points) ++tied;
  if (tied == 1) return;
  // Only plays that empty the bag: the spread of one that leaves tiles would
  // sum over many opponent lines.
  std::vector<size_t> idx;
  for (size_t i = 0; i < tied; ++i)
    if (plays[i].move.num_glyphs() >= num_in_bag_) idx.push_back(i);
  if (idx.empty()) return;
  if (idx.size() == 1) {
    std::swap(plays[0], plays[idx[0]]);
    return;
  }
  stable_sort_desc(idx, [&](size_t i) { return int(plays[i].move.score()); });
  if (idx.size() > size_t(PreEndgameSolver::kTiebreakPlays))
    idx.resize(PreEndgameSolver::kTiebreakPlays);
  const std::vector<Perm> perms = permutations(maybe_in_bag, num_in_bag_);
  for (size_t i : idx) {
    for (const Perm& p : perms) {
      PegGame g = root;
      g.stage(unseen_, p.tiles);
      recursive_solve(plays[i], plays[i].move, {p.tiles, p.count}, g, true, true, 0, true);
    }
  }
  // Plays without a spread sink below those with one.
  std::stable_sort(plays.begin(), plays.end(), [](const PegPlay& a, const PegPlay& b) {
    if (a.spread_set != b.spread_set) return a.spread_set;
    return a.spread > b.spread;
  });
}

// The mover's plays and pass, best static equity first.
std::vector<Move> root_moves(const PreEndgamePosition& pos, int num_in_bag) {
  static const Rack kNoOpp;
  const MoveRequest req{pos.board,    *pos.dict,     pos.my_rack, kNoOpp,
                        pos.my_score, pos.opp_score, num_in_bag};
  std::vector<Move> moves = generate_legal_plays(req);
  moves.push_back(Move::pass());
  const HastyEquity& eq = HastyEquity::instance();
  std::vector<std::pair<double, Move>> ranked;
  for (const Move& m : moves)
    ranked.push_back({eq.equity(m, pos.board, num_in_bag, kNoOpp, pos.my_rack), m});
  stable_sort_desc(ranked, [](const auto& r) { return r.first; });
  for (size_t i = 0; i < moves.size(); ++i) moves[i] = ranked[i].second;
  return moves;
}

}  // namespace

void PreEndgameSolver::Params::add_options(boost::program_options::options_description& desc,
                                           const std::string& prefix) {
  namespace po = boost::program_options;
  const std::string unproven_flag = prefix + "unproven";
  // Parsed through a string, so the enum needs no stream operators.
  const auto set_unproven = [this, unproven_flag](const std::string& name) {
    if (name == "spread") {
      unproven = UnprovenPolicy::kEstimatedSpread;
    } else if (name == "loss") {
      unproven = UnprovenPolicy::kLoss;
    } else {
      throw util::CleanException("--{} must be 'spread' or 'loss', got '{}'", unproven_flag, name);
    }
  };
  desc.add_options()  //
    ((prefix + "max-effort").c_str(), po::value<int>(&max_effort)->default_value(max_effort),
     "highest endgame effort (for our solver, its depth cap) the pre-endgame deepens to; 0 "
     "follows Macondo's schedule, 2 to 7 by how far apart the scores are")  //
    (unproven_flag.c_str(),
     po::value<std::string>()
       ->default_value(unproven == UnprovenPolicy::kLoss ? "loss" : "spread")
       ->notifier(set_unproven),
     "how an endgame the solver cannot classify is scored: 'spread' (by the sign of its "
     "estimate) or 'loss'");
}

int PreEndgameSolver::macondo_max_effort(int spread) {
  const int s = std::abs(spread);
  if (s >= 100) return 2;
  if (s >= 80) return 3;
  if (s >= 60) return 4;
  if (s >= 50) return 5;
  return 7;
}

std::vector<PreEndgameSolver::RankedPlay> PreEndgameSolver::solve(const PreEndgamePosition& pos,
                                                                  const Params& params) {
  const TileCounts unseen = pos.board.unseen_tiles(pos.my_rack);
  const int num_in_bag = unseen.size() - RACK_SIZE;
  if (num_in_bag < 1 || num_in_bag > kMaxInBag)
    throw util::Exception("PreEndgameSolver: the bag must hold 1 to {} tiles, not {}", kMaxInBag,
                          num_in_bag);
  TileCounts maybe_in_bag = unseen;
  for (int i = 0; i < pos.opp_known.size(); ++i) maybe_in_bag.remove(pos.opp_known.tiles()[i]);

  PegGame root;
  root.board = pos.board;
  root.board.ensure_movegen_caches(*pos.dict);
  root.racks[0] = pos.my_rack;
  root.scores = {pos.my_score, pos.opp_score};
  // Macondo's endgame mode keeps a pending scoreless turn only if the next
  // would have ended the game under the six-turn rule.
  root.scoreless = pos.scoreless_turns + 1 >= 6 ? 1 : 0;

  const int max_effort =
    params.max_effort > 0 ? params.max_effort : macondo_max_effort(pos.my_score - pos.opp_score);
  std::vector<Move> order = root_moves(pos, num_in_bag);
  std::vector<PegPlay> plays;
  for (int effort = 1; effort <= max_effort; ++effort) {
    PegSearch search(*pos.dict, oracle_, params.unproven, num_in_bag, unseen, effort);
    plays.assign(order.size(), PegPlay{});
    for (size_t i = 0; i < order.size(); ++i) plays[i].move = order[i];
    for (PegPlay& p : plays) search.process_play(p, root, maybe_in_bag);
    stable_sort_desc(plays, [](const PegPlay& p) { return p.points; });
    search.tiebreak(plays, root, maybe_in_bag);
    for (size_t i = 0; i < plays.size(); ++i) order[i] = plays[i].move;
  }
  std::vector<RankedPlay> out;
  for (const PegPlay& p : plays) out.push_back({p.move, p.points, p.total()});
  return out;
}

}  // namespace scribblez
