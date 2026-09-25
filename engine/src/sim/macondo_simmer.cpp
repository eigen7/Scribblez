#include "sim/macondo_simmer.h"

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "game/bag.h"
#include "game/board.h"
#include "game/rack.h"
#include "lexicon/hasty_equity.h"
#include "sim/win_pct_table.h"
#include "util/assert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <thread>

namespace scribblez {

namespace {

constexpr int kMaxScorelessTurns = 6;

// A rollout's game, played by the rules of Macondo's Game in simulation mode,
// which match Game::play_loop's: going out collects twice the opponent's rack,
// and six scoreless turns end the game with each side docked its own rack.
struct RolloutGame {
  Board board;
  std::array<Rack, 2> racks;
  Bag bag;
  std::array<int, 2> scores;
  int scoreless = 0;
  int on_turn = 0;
  bool over = false;

  // Plays `m` for the side to move, draws its replacements, and passes the
  // turn unless the game ended.
  void play(const Move& m);

 private:
  void refill(int p);
};

void RolloutGame::refill(int p) {
  while (racks[p].size() < RACK_SIZE && bag.size() > 0) racks[p].add(bag.draw());
}

void RolloutGame::play(const Move& m) {
  Rack& rack = racks[on_turn];
  for (int i = 0; i < m.num_glyphs(); ++i) {
    const bool ok = rack.remove(m.glyph(i).rack_tile());
    RELEASE_ASSERT(ok);
  }
  if (m.type() == MoveType::PLAY) {
    board.apply(m);
    scores[on_turn] += m.score();
    scoreless = 0;
    refill(on_turn);
    if (rack.empty()) {
      scores[on_turn] += 2 * racks[1 - on_turn].point_value();
      over = true;
      return;
    }
  } else {
    if (m.type() == MoveType::EXCHANGE) {
      // Draw before the surrendered tiles rejoin the bag, as Macondo's
      // Bag.Exchange does.
      refill(on_turn);
      for (int i = 0; i < m.num_glyphs(); ++i) bag.put_back(m.glyph(i).rack_tile());
    }
    ++scoreless;
  }
  if (scoreless >= kMaxScorelessTurns) {
    for (int p = 0; p < 2; ++p) scores[p] -= racks[p].point_value();
    over = true;
    return;
  }
  on_turn = 1 - on_turn;
}

// What an iteration shares across its candidates: the opponent's rack (their
// known tiles plus draws) and the bag left after dealing it.
struct Deal {
  Rack opp_rack;
  Bag bag;
};

Deal deal(const SimPosition& pos, const TileCounts& pool, uint64_t seed) {
  Deal d{pos.opp_leave, Bag(seed, pool)};
  for (int i = 0; i < pos.opp_leave.size(); ++i) d.bag.remove(pos.opp_leave.tiles()[i]);
  while (d.opp_rack.size() < RACK_SIZE && d.bag.size() > 0) d.opp_rack.add(d.bag.draw());
  return d;
}

Rack leave_after(Rack rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  return rack;
}

// A rollout's win probability for the mover. A finished game, or one with no
// tile unseen by the mover, is decided by its spread alone. Otherwise the
// table reads the spread plus leftover leave values (rounded) from the point
// of view of the side on turn, which after an even number of plies is the
// opponent: the spread is flipped going in and the probability coming out.
double rollout_win_prob(int spread, double leftover, bool over, int tiles_unseen, int plies) {
  if (over || tiles_unseen == 0) return spread > 0 ? 1.0 : spread == 0 ? 0.5 : 0.0;
  const bool opp_on_turn = plies % 2 == 0;
  int s = spread + int(std::round(leftover));
  if (opp_on_turn) s = -s;
  float p = WinPctTable::macondo_default().win_prob(s, tiles_unseen);
  if (opp_on_turn) p = 1.0f - p;
  return p;
}

struct RolloutOutcome {
  double win_prob = 0.0;
  double equity = 0.0;
};

// One candidate's rollout: the candidate, then `plies` plies of each side's
// best static-equity move. The last two plies' leaves are valued (the mover's
// for, the opponent's against), standing in for the tiles they carry forward.
RolloutOutcome rollout(const Dictionary& dict, const SimPosition& pos, const Move& candidate,
                       const Deal& d, int scoreless_turns, int plies) {
  const int mover = pos.mover, opp = 1 - pos.mover;
  RolloutGame g{pos.board, {}, d.bag, pos.scores, scoreless_turns, mover};
  g.racks[mover] = pos.rack;
  g.racks[opp] = d.opp_rack;
  g.play(candidate);
  double leftover = 0.0;
  for (int ply = 0; ply < plies && !g.over; ++ply) {
    const int on = g.on_turn;
    const MoveRequest req{g.board,          dict,        g.racks[on], g.racks[1 - on], g.scores[on],
                          g.scores[1 - on], g.bag.size()};
    const Move best = hasty_best_move_wmp(req);
    if (ply >= plies - 2) {
      const double lv = HastyEquity::instance().leave_value(leave_after(g.racks[on], best));
      leftover += on == mover ? lv : -lv;
    }
    g.play(best);
  }
  const int spread = g.scores[mover] - g.scores[opp];
  const int initial_spread = pos.scores[mover] - pos.scores[opp];
  const int tiles_unseen = g.bag.size() + g.racks[opp].size();
  return {rollout_win_prob(spread, leftover, g.over, tiles_unseen, plies),
          double(spread - initial_spread) + leftover};
}

// Everything one batch of iterations reads.
struct Batch {
  const Dictionary& dict;
  const SimPosition& pos;
  const TileCounts& pool;
  const std::vector<SimmedPlay>& plays;
  const std::vector<int>& active;  // indices of the unpruned plays
  int scoreless_turns;
  int plies;
  uint64_t first_seed;  // iteration i of the batch is seeded by first_seed + i
  int iterations;
};

// Worker t runs iterations t, t + threads, ... of the batch into `out`, where
// iteration i's rollout of active play j lands at [i * active.size() + j].
// Workers write disjoint slots, so they need no synchronization.
void run_batch_worker(const Batch& b, int t, int threads, std::vector<RolloutOutcome>& out) {
  for (int i = t; i < b.iterations; i += threads) {
    const Deal d = deal(b.pos, b.pool, b.first_seed + uint64_t(i));
    for (size_t j = 0; j < b.active.size(); ++j) {
      out[size_t(i) * b.active.size() + j] =
        rollout(b.dict, b.pos, b.plays[size_t(b.active[j])].move, d, b.scoreless_turns, b.plies);
    }
  }
}

// Thread entry: captures any exception into `err` for the joining thread to
// rethrow, since one escaping a std::thread calls std::terminate.
void batch_worker(const Batch& b, int t, int threads, std::vector<RolloutOutcome>* out,
                  std::exception_ptr* err) {
  try {
    run_batch_worker(b, t, threads, *out);
  } catch (...) {
    *err = std::current_exception();
  }
}

std::vector<RolloutOutcome> run_batch(const Batch& b, int threads) {
  std::vector<RolloutOutcome> out(size_t(b.iterations) * b.active.size());
  threads = std::clamp(threads, 1, b.iterations);
  if (threads == 1) {
    run_batch_worker(b, 0, 1, out);
    return out;
  }
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(threads);
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(batch_worker, std::cref(b), t, threads, &out, &errors[t]);
  for (std::thread& w : workers) w.join();
  for (const std::exception_ptr& e : errors)
    if (e) std::rethrow_exception(e);
  return out;
}

std::vector<int> unpruned(const std::vector<SimmedPlay>& plays) {
  std::vector<int> active;
  for (int i = 0; i < int(plays.size()); ++i)
    if (!plays[i].ignored) active.push_back(i);
  return active;
}

// Folds a batch into the statistics in iteration order, the order a
// single-threaded Macondo sim pushes them in.
void accumulate(const std::vector<RolloutOutcome>& out, const std::vector<int>& active,
                std::vector<SimmedPlay>& plays) {
  for (size_t k = 0; k < out.size(); ++k) {
    SimmedPlay& p = plays[size_t(active[k % active.size()])];
    p.win_prob.push(out[k].win_prob);
    p.equity.push(out[k].equity);
  }
}

}  // namespace

MacondoSimmer::Result MacondoSimmer::simulate(const Dictionary& dict, const SimPosition& pos,
                                              const std::vector<Move>& candidates,
                                              int scoreless_turns, const Params& params,
                                              uint64_t seed) {
  Result result;
  for (const Move& m : candidates) result.plays.emplace_back().move = m;
  // With one candidate there is nothing to compare; Macondo would sim a batch
  // and stop at the first check.
  if (candidates.size() < 2) return result;

  const TileCounts pool = pos.board.unseen_tiles(pos.rack);
  RELEASE_ASSERT(pool.size() > RACK_SIZE, "MacondoSimmer needs tiles in the bag");
  const MacondoAutoStopper stopper(pos.board);
  while (true) {
    int batch = MacondoAutoStopper::kCheckInterval;
    if (params.max_iterations > 0)
      batch = std::min<uint64_t>(batch, uint64_t(params.max_iterations) - result.iterations);
    const std::vector<int> active = unpruned(result.plays);
    const Batch b{dict,         pos,
                  pool,         result.plays,
                  active,       scoreless_turns,
                  params.plies, seed + result.iterations,
                  batch};
    accumulate(run_batch(b, params.threads), active, result.plays);
    result.iterations += uint64_t(batch);
    if (params.max_iterations > 0 && result.iterations >= uint64_t(params.max_iterations)) break;
    if (stopper.should_stop(result.iterations, result.plays, params.plies)) break;
  }
  rank_simmed_plays(result.plays);
  return result;
}

}  // namespace scribblez
