// endgame_bench: measures what the endgame solver costs and what it buys.
// Results and methodology are written up in docs/endgame_bench_results.md.
//
//   endgame_bench --mode=endgames --games 400 --budgets 100,220,1600 --threads 16
//   endgame_bench --mode=games --games 1000 --budgets 220,1600 --threads 16
//
// --mode=endgames (default) is a margin sweep. It plays --games
// HastyBot-vs-HastyBot games and captures each one's first bag-empty position.
// It then replays every captured endgame with EndgameHastyBot on the side to
// move and plain HastyBot replying, at every (start-of-endgame margin, budget)
// pair. Only the spread matters to either agent, so a margin m is set as scores
// (m, 0). Two tables come out, margins as rows and budgets as columns:
//   - skill: the solver seat's mean game value (win 1, draw 0.5, loss 0) minus
//     that of a HastyBot-vs-HastyBot playout, in win% points;
//   - cost: a self-play game's time with the endgame solved, as a multiple of a
//     hasty-vs-hasty game. Only the first --time-games games are timed, on one
//     thread with nothing else running.
//
// --mode=games times --games full games of hasty-vs-hasty and of
// endgame-vs-endgame at each budget, on the same seeds, and reports the time
// ratios. It then plays EndgameHastyBot against HastyBot with seats mirrored
// per seed and reports win% and W/D/L, bucketed by the seed's
// hasty-vs-hasty spread when the bag empties (--spread-buckets).
//
// All games respect projections (a proven endgame ends at its certificate), as
// self-play generation does; --projections 0 turns that off in endgames mode.

#include "agent/agent.h"
#include "agent/endgame_hasty_bot.h"
#include "agent/endgame_turn_policy.h"
#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"
#include "game/bag.h"
#include "game/board.h"
#include "game/game.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/misc.h"
#include "util/string.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace scribblez {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Run body(worker, item) for every item in [0, items) on `threads` workers.
// Items are handed out one at a time because sweep cost varies by orders of
// magnitude between positions: static chunks would leave most workers waiting
// on the slowest one.
void parallel_for(int items, int threads, const std::function<void(int worker, int item)>& body) {
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (int i = next.fetch_add(1); i < items; i = next.fetch_add(1)) body(t, i);
    });
  }
  for (std::thread& th : pool) th.join();
}

// The first bag-empty decision point of one HastyBot self-play game: enough to
// replay the endgame with either agent type. my_rack belongs to the side to
// move.
struct CapturedEndgame {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless = 0;
};

// A HastyBot that records the first bag-empty position it is asked to move on.
// It counts consecutive scoreless turns the way the game loop does, since the
// solver takes that count as input.
class FirstEndgameCapturer : public Agent {
 public:
  FirstEndgameCapturer(int thread_id, const std::string& name, CapturedEndgame& sink,
                       bool& captured)
      : Agent(thread_id, name),
        bot_({.thread_id = thread_id, .name = name}),
        sink_(sink),
        captured_(captured) {}

  MoveDecision make_move(const MoveRequest& req) override {
    if (req.bag_size == 0 && !captured_) {
      captured_ = true;
      sink_.board = req.board;
      sink_.my_rack = req.my_rack;
      sink_.opp_rack = req.opp_rack;
      sink_.my_score = req.my_score;
      sink_.opp_score = req.opp_score;
      sink_.scoreless = scoreless_;
    }
    return bot_.make_move(req);
  }

  void observe_move(const Move& move) override {
    if (move.type() == MoveType::PLAY)
      scoreless_ = 0;
    else
      ++scoreless_;
  }

  void begin_game(const BeginGameRequest& /*req*/) override { scoreless_ = 0; }

 private:
  HastyBotAgent bot_;
  CapturedEndgame& sink_;
  bool& captured_;
  int scoreless_ = 0;
};

std::vector<uint64_t> parse_budgets(const std::string& csv) {
  std::vector<uint64_t> out;
  for (const std::string& tok : util::split(csv, ',')) out.push_back(std::stoull(tok));
  return out;
}

std::vector<int> parse_thresholds(const std::string& csv) {
  std::vector<int> out;
  for (const std::string& tok : util::split(csv, ',')) out.push_back(std::stoi(tok));
  return out;
}

// Bucket k holds |spread| in [t_{k-1}, t_k); the last bucket is unbounded.
int bucket_of(int abs_spread, const std::vector<int>& thresholds) {
  int k = 0;
  while (k < int(thresholds.size()) && abs_spread >= thresholds[k]) ++k;
  return k;
}

std::string bucket_label(int k, const std::vector<int>& thresholds) {
  const int lo = k == 0 ? 0 : thresholds[k - 1];
  if (k == int(thresholds.size())) return std::format("{}+", lo);
  return std::format("{}-{}", lo, thresholds[k] - 1);
}

// Play `games` HastyBot-vs-HastyBot games seeded base_seed+i and return the
// first bag-empty position of each game that reached one, in seed order.
std::vector<CapturedEndgame> capture_endgames(const Dictionary& dict, uint64_t base_seed, int games,
                                              int threads) {
  std::vector<CapturedEndgame> caps(games);
  std::vector<char> captured(games, 0);
  parallel_for(games, threads, [&](int worker, int i) {
    bool got = false;
    FirstEndgameCapturer a0(worker, "A", caps[i], got);
    FirstEndgameCapturer a1(worker, "B", caps[i], got);
    Game g(a0, a1, dict, base_seed + uint64_t(i));
    g.play();
    captured[i] = got ? 1 : 0;
  });
  std::vector<CapturedEndgame> out;
  for (int i = 0; i < games; ++i) {
    if (captured[i]) out.push_back(caps[i]);
  }
  return out;
}

// The captured endgame's bag, which is empty: the board and both racks account
// for every tile. play_from then draws nothing, so the racks stay as captured.
Bag empty_pool(const CapturedEndgame& cap) {
  Bag pool(/*seed=*/1);
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = cap.board.at(r, c);
      if (g.has_letter()) pool.remove(g.rack_tile());
    }
  }
  for (int i = 0; i < cap.my_rack.size(); ++i) pool.remove(cap.my_rack.tiles()[i]);
  for (int i = 0; i < cap.opp_rack.size(); ++i) pool.remove(cap.opp_rack.tiles()[i]);
  return pool;
}

// Mean wall time of a HastyBot-vs-HastyBot game over the timed games' seeds:
// the cost table's unit, a self-play game with no endgame solving.
double hasty_game_ms(const Dictionary& dict, uint64_t base_seed, int games) {
  HastyBotAgent a0(HastyBotAgent::Params{.thread_id = 0, .name = "A"});
  HastyBotAgent a1(HastyBotAgent::Params{.thread_id = 0, .name = "B"});
  const auto t0 = Clock::now();
  for (int i = 0; i < games; ++i) {
    Game g(a0, a1, dict, base_seed + uint64_t(i));
    g.set_respect_projections(true);
    g.play();
  }
  return 1000.0 * seconds_since(t0) / games;
}

// Game value for the side with this final spread: win 1, draw 0.5, loss 0.
double win_fraction(int spread) {
  if (spread > 0) return 1.0;
  if (spread < 0) return 0.0;
  return 0.5;
}

// --- Margin sweep -----------------------------------------------------------

// One solver-seat playout: the side to move's final spread (deterministic), and
// the solver's wall time (meaningful only in the single-threaded timed phase).
struct SolverOutcome {
  int spread = 0;
  uint64_t solve_ns = 0;
};

// The same, with the solver's full totals; the budget-nesting skip in
// sweep_column needs max_solve_nodes.
struct SolverPlayout {
  int spread = 0;
  EndgameTurnPolicy::SolveTotals totals;
};

// Play one captured endgame with EndgameHastyBot to move (scores {margin, 0})
// and HastyBot replying. Agents are rebuilt per call with a fixed thread_id and
// name, so HastyBot's deterministic tie-breaks make the spread a pure function
// of position, margin, and budget, whichever worker runs it. `incremental`
// toggles the solver's incremental move-list maintenance, which changes speed
// but never results.
SolverPlayout run_solver_playout(const Dictionary& dict, const CapturedEndgame& cap, int margin,
                                 const EndgameSolver::Params& params, int thread_id,
                                 bool incremental, bool projections) {
  EndgameHastyBotAgent::Params ep;
  ep.hasty = HastyBotAgent::Params{.thread_id = thread_id, .name = "EndgameHastyBot"};
  ep.solver = params;
  EndgameHastyBotAgent eg(ep);
  eg.endgame().set_incremental_movegen(incremental);
  HastyBotAgent opp(HastyBotAgent::Params{.thread_id = thread_id, .name = "HastyBot"});

  const Bag pool = empty_pool(cap);
  Game g(eg, opp, dict, /*seed=*/1);
  g.set_respect_projections(projections);
  g.play_from(cap.board, {margin, 0}, {cap.my_rack, cap.opp_rack}, pool, /*to_move=*/0);

  SolverPlayout out;
  out.spread = g.score(0) - g.score(1);
  out.totals = eg.endgame().solve_totals();
  return out;
}

// The side to move's spread gain over a HastyBot-vs-HastyBot playout of the
// captured endgame. HastyBot never reads scores, so this one number gives the
// baseline at every margin: the final spread at margin m is m + delta.
int baseline_delta(const Dictionary& dict, const CapturedEndgame& cap, int thread_id) {
  HastyBotAgent a0(HastyBotAgent::Params{.thread_id = thread_id, .name = "A"});
  HastyBotAgent a1(HastyBotAgent::Params{.thread_id = thread_id, .name = "B"});
  const Bag pool = empty_pool(cap);
  Game g(a0, a1, dict, /*seed=*/1);
  g.play_from(cap.board, {0, 0}, {cap.my_rack, cap.opp_rack}, pool, /*to_move=*/0);
  return g.score(0) - g.score(1);
}

std::vector<int> margin_axis(int margin_min, int margin_max, int margin_step) {
  std::vector<int> margins;
  for (int m = margin_min; m <= margin_max; m += margin_step) margins.push_back(m);
  return margins;
}

// Fill one (game, margin) column of the grid, every budget at that margin,
// into `grid[cell_base + budget index]`. A column rather than a whole game is
// the unit of parallel work because per-position solve cost spans orders of
// magnitude.
//
// Budgets run in descending order so the budget-nesting skip can reuse a larger
// budget's result. If a run's max_solve_nodes is <= a smaller budget b', rerunning
// at b' is bit-identical: no solve hit the larger cap, and a solve declined for
// having more root moves than the larger budget stays declined at b'. The skip
// is unsound under spread_matters, whose class pass gets half the budget, so
// the result depends on the budget value itself. `skipped` counts the cells
// filled without a playout.
void sweep_column(const Dictionary& dict, const CapturedEndgame& cap, int margin,
                  const std::vector<std::pair<uint64_t, int>>& desc_budgets,
                  EndgameSolver::Params params, bool incremental, bool projections, int thread_id,
                  size_t cell_base, std::vector<SolverOutcome>& grid,
                  std::atomic<uint64_t>& skipped) {
  const bool skip_enabled = !params.spread_matters;
  SolverOutcome last;
  uint64_t last_max_nodes = 0;
  bool have_last = false;
  uint64_t skips = 0;
  for (const std::pair<uint64_t, int>& bd : desc_budgets) {
    if (skip_enabled && have_last && last_max_nodes <= bd.first) {
      grid[cell_base + bd.second] = last;
      ++skips;
      continue;
    }
    params.budget = bd.first;
    const SolverPlayout p =
      run_solver_playout(dict, cap, margin, params, thread_id, incremental, projections);
    last = {p.spread, p.totals.solve_ns};
    last_max_nodes = p.totals.max_solve_nodes;
    have_last = true;
    grid[cell_base + bd.second] = last;
  }
  skipped.fetch_add(skips, std::memory_order_relaxed);
}

std::string join_budgets(const std::vector<uint64_t>& budgets) {
  std::string s;
  for (size_t i = 0; i < budgets.size(); ++i) {
    if (i) s += ',';
    s += std::to_string(budgets[i]);
  }
  return s;
}

// Print the skill table over every game (with a trailing baseline win% column)
// and the cost table over the first `timed_games`. Cost is a self-play game's
// time with the endgame solved relative to `baseline_ms`, the same unit as
// games mode's throughput ratios; 1.00x means the solver is free.
void print_sweep_tables(const std::vector<int>& margins, const std::vector<uint64_t>& budgets,
                        const std::vector<int>& d0, const std::vector<SolverOutcome>& grid,
                        int timed_games, double baseline_ms) {
  const int ms = margins.size();
  const int bs = budgets.size();
  const int games = d0.size();

  std::printf(
    "\nskill: solver win%% minus hasty win%% (first actor's seat), by margin x budget:\n");
  std::printf("%8s", "margin");
  for (uint64_t b : budgets) std::printf(" %10llu", static_cast<unsigned long long>(b));
  std::printf(" %12s\n", "hasty win%");
  for (int mi = 0; mi < ms; ++mi) {
    std::printf("%8d", margins[mi]);
    double base_win = 0;
    for (int g = 0; g < games; ++g) base_win += win_fraction(margins[mi] + d0[g]);
    base_win /= games;
    for (int bi = 0; bi < bs; ++bi) {
      double skill = 0;
      for (int g = 0; g < games; ++g) {
        const SolverOutcome& o = grid[(size_t(g) * ms + mi) * bs + bi];
        skill += win_fraction(o.spread) - win_fraction(margins[mi] + d0[g]);
      }
      std::printf(" %+10.1f", 100.0 * skill / games);
    }
    std::printf(" %11.1f\n", 100.0 * base_win);
  }

  if (timed_games == 0) return;
  std::printf("\ncost: self-play game time as a multiple of hasty-vs-hasty, by margin x budget:\n");
  std::printf("%8s", "margin");
  for (uint64_t b : budgets) std::printf(" %10llu", static_cast<unsigned long long>(b));
  std::printf("\n");
  for (int mi = 0; mi < ms; ++mi) {
    std::printf("%8d", margins[mi]);
    for (int bi = 0; bi < bs; ++bi) {
      double ns = 0;
      for (int g = 0; g < timed_games; ++g) {
        ns += grid[(size_t(g) * ms + mi) * bs + bi].solve_ns;
      }
      std::printf(" %10.3f", 1.0 + ns / timed_games / 1e6 / baseline_ms);
    }
    std::printf("\n");
  }
}

// Sweep every captured endgame over the margin x budget grid. The first
// `time_games` games run alone on one thread and are the only ones the cost
// table reads: solve times are only trustworthy with nothing else competing for
// cores, caches, and clock. The rest run on `threads` workers and add their
// deterministic spreads to the skill table, where sample size buys accuracy.
void run_endgames_mode(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                       const std::vector<uint64_t>& budgets, EndgameSolver::Params params,
                       bool incremental, bool projections, int margin_min, int margin_max,
                       int margin_step, int time_games) {
  const std::vector<CapturedEndgame> caps = capture_endgames(dict, base_seed, games, threads);
  const int g_count = caps.size();
  const std::vector<int> margins = margin_axis(margin_min, margin_max, margin_step);
  const int ms = margins.size();
  const int bs = budgets.size();
  const int timed = std::min(time_games, g_count);

  std::printf(
    "endgames mode (margin sweep): %d games, %d captured, margins %d..%d step %d, budgets=%s, "
    "spread-matters=%d, incremental=%d, projections=%d, threads=%d, timed games=%d\n",
    games, g_count, margin_min, margin_max, margin_step, join_budgets(budgets).c_str(),
    params.spread_matters ? 1 : 0, incremental ? 1 : 0, projections ? 1 : 0, threads, timed);
  if (g_count == 0) {
    std::printf("no game reached an endgame; nothing to sweep\n");
    return;
  }

  std::vector<std::pair<uint64_t, int>> desc_budgets;
  for (int bi = 0; bi < bs; ++bi) desc_budgets.emplace_back(budgets[bi], bi);
  std::sort(desc_budgets.begin(), desc_budgets.end(),
            [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) {
              return a.first > b.first;
            });

  std::vector<int> d0(g_count, 0);
  parallel_for(g_count, threads,
               [&](int worker, int g) { d0[g] = baseline_delta(dict, caps[g], worker); });

  std::vector<SolverOutcome> grid(size_t(g_count) * ms * bs);
  std::atomic<uint64_t> skipped{0};
  const auto sweep = [&](int worker, int g, int mi) {
    sweep_column(dict, caps[g], margins[mi], desc_budgets, params, incremental, projections, worker,
                 (size_t(g) * ms + mi) * bs, grid, skipped);
  };

  // The baseline shares the timed phase's conditions: one thread, nothing else
  // running, so its milliseconds are comparable with the sweep's.
  const double baseline_ms = timed > 0 ? hasty_game_ms(dict, base_seed, timed) : 0.0;
  const auto t_timed = Clock::now();
  for (int g = 0; g < timed; ++g) {
    for (int mi = 0; mi < ms; ++mi) sweep(/*worker=*/0, g, mi);
  }
  const double timed_s = seconds_since(t_timed);
  const auto t_rest = Clock::now();
  parallel_for((g_count - timed) * ms, threads,
               [&](int worker, int i) { sweep(worker, timed + i / ms, i % ms); });
  std::printf("swept %d games in %.1f s single-threaded, %d in %.1f s on %d threads\n", timed,
              timed_s, g_count - timed, seconds_since(t_rest), threads);
  if (timed > 0)
    std::printf("hasty-vs-hasty baseline: %.3f ms/game over %d games, single-threaded\n",
                baseline_ms, timed);

  const uint64_t total_cells = uint64_t(g_count) * ms * bs;
  std::printf("budget-nesting skip: %llu of %llu solver playouts avoided\n",
              static_cast<unsigned long long>(skipped.load()),
              static_cast<unsigned long long>(total_cells));

  print_sweep_tables(margins, budgets, d0, grid, timed, baseline_ms);
}

// --- Games mode -------------------------------------------------------------

using AgentFactory = std::function<std::unique_ptr<Agent>(int thread_id)>;

// Play `games` games (seed base_seed+i) between factory-built agents and return
// the wall-clock seconds. Each thread takes a contiguous chunk of seeds and
// builds its agent pair once.
double run_config(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                  const AgentFactory& make0, const AgentFactory& make1) {
  const auto t0 = Clock::now();
  std::vector<std::thread> pool;
  const int per = (games + threads - 1) / threads;
  for (int t = 0; t < threads; ++t) {
    const int lo = t * per;
    const int hi = std::min(games, lo + per);
    if (lo >= hi) break;
    pool.emplace_back([&, t, lo, hi] {
      std::unique_ptr<Agent> p0 = make0(t);
      std::unique_ptr<Agent> p1 = make1(t);
      for (int i = lo; i < hi; ++i) {
        Game g(*p0, *p1, dict, base_seed + uint64_t(i));
        g.set_respect_projections(true);
        g.play();
      }
    });
  }
  for (std::thread& th : pool) th.join();
  return seconds_since(t0);
}

AgentFactory hasty_factory() {
  return [](int tid) -> std::unique_ptr<Agent> {
    return std::make_unique<HastyBotAgent>(
      HastyBotAgent::Params{.thread_id = tid, .name = "HastyBot"});
  };
}

AgentFactory endgame_factory(const EndgameSolver::Params& params, bool incremental) {
  return [params, incremental](int tid) -> std::unique_ptr<Agent> {
    EndgameHastyBotAgent::Params p;
    p.hasty = HastyBotAgent::Params{.thread_id = tid, .name = "EndgameHastyBot"};
    p.solver = params;
    auto agent = std::make_unique<EndgameHastyBotAgent>(p);
    agent->endgame().set_incremental_movegen(incremental);
    return agent;
  };
}

// EndgameHastyBot's record against HastyBot in one spread bucket.
struct H2H {
  int games = 0;
  int wins = 0;
  int draws = 0;
  int losses = 0;
};

// |spread| when the bag first empties in this seed's HastyBot-vs-HastyBot game,
// or -1 if it never does. It depends only on the seed, so every configuration
// and both seat orders bucket a seed the same way.
int baseline_bag_empty_spread(const Dictionary& dict, uint64_t seed) {
  CapturedEndgame cap;
  bool captured = false;
  FirstEndgameCapturer a0(0, "A", cap, captured);
  FirstEndgameCapturer a1(0, "B", cap, captured);
  Game g(a0, a1, dict, seed);
  g.play();
  if (!captured) return -1;
  return std::abs(cap.my_score - cap.opp_score);
}

// Play EndgameHastyBot against HastyBot over ceil(games / 2) seeds, bucketed by
// baseline_bag_empty_spread; the last bucket holds seeds whose bag never
// empties. Each seed is played twice with seats mirrored, so tile-draw luck
// (who gets the blanks) cancels instead of dominating the variance. Runs
// single-threaded so the results are deterministic in `base_seed`.
std::vector<H2H> endgame_vs_hasty(const Dictionary& dict, uint64_t base_seed, int games,
                                  const EndgameSolver::Params& params, bool incremental,
                                  const std::vector<int>& thresholds) {
  const AgentFactory eg = endgame_factory(params, incremental);
  const AgentFactory hb = hasty_factory();
  std::vector<H2H> buckets(thresholds.size() + 2);
  for (int i = 0; i < (games + 1) / 2; ++i) {
    const uint64_t seed = base_seed + uint64_t(i);
    const int be = baseline_bag_empty_spread(dict, seed);
    H2H& h = buckets[be < 0 ? buckets.size() - 1 : size_t(bucket_of(be, thresholds))];
    for (int eg_seat = 0; eg_seat < 2; ++eg_seat) {
      std::unique_ptr<Agent> p0 = eg_seat == 0 ? eg(0) : hb(0);
      std::unique_ptr<Agent> p1 = eg_seat == 0 ? hb(0) : eg(0);
      Game g(*p0, *p1, dict, seed);
      g.set_respect_projections(true);
      g.play();
      const int spread = g.score(eg_seat) - g.score(1 - eg_seat);
      ++h.games;
      if (spread > 0)
        ++h.wins;
      else if (spread < 0)
        ++h.losses;
      else
        ++h.draws;
    }
  }
  return buckets;
}

void print_games_row(const char* config, const std::string& budget, double total_s, int games,
                     double ratio) {
  std::printf("%-22s %11s %10.3f %10.4f %10.3f %9.2fx\n", config, budget.c_str(), total_s,
              total_s / games, games / total_s, ratio);
}

void print_h2h_row(uint64_t budget, const std::string& label, const H2H& h) {
  const double winpct = 100.0 * (h.wins + 0.5 * h.draws) / h.games;
  std::printf("%11llu %9s %8d %9.1f%% %8d %8d %8d\n", static_cast<unsigned long long>(budget),
              label.c_str(), h.games, winpct, h.wins, h.draws, h.losses);
}

void run_games_mode(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                    const std::vector<uint64_t>& budgets, EndgameSolver::Params params,
                    bool incremental, const std::vector<int>& thresholds) {
  std::printf(
    "games mode: %d games/config, threads=%d, plies=%d, spread-matters=%d, "
    "incremental=%d\n\n",
    games, threads, params.plies, params.spread_matters ? 1 : 0, incremental ? 1 : 0);
  std::printf("%-22s %11s %10s %10s %10s %10s\n", "config", "budget", "total s", "s/game",
              "games/s", "ratio");

  const double base_s =
    run_config(dict, base_seed, games, threads, hasty_factory(), hasty_factory());
  print_games_row("hasty-vs-hasty", "-", base_s, games, 1.0);

  for (uint64_t b : budgets) {
    params.budget = b;
    const AgentFactory f = endgame_factory(params, incremental);
    const double s = run_config(dict, base_seed, games, threads, f, f);
    print_games_row("endgame-vs-endgame", std::to_string(b), s, games, s / base_s);
  }

  // Strength shows in the small-spread buckets, where the endgame still decides
  // the game.
  std::printf("\nhead-to-head (endgame-vs-hasty, mirrored seats, by baseline bag-empty spread):\n");
  std::printf("%11s %9s %8s %10s %8s %8s %8s\n", "budget", "|spread|", "games", "win%", "W", "D",
              "L");
  for (uint64_t b : budgets) {
    params.budget = b;
    const std::vector<H2H> buckets =
      endgame_vs_hasty(dict, base_seed, games, params, incremental, thresholds);
    H2H total;
    for (size_t k = 0; k < buckets.size(); ++k) {
      const H2H& h = buckets[k];
      total.games += h.games;
      total.wins += h.wins;
      total.draws += h.draws;
      total.losses += h.losses;
      if (h.games == 0) continue;
      const std::string label = k + 1 == buckets.size() ? "none" : bucket_label(int(k), thresholds);
      print_h2h_row(b, label, h);
    }
    print_h2h_row(b, "all", total);
  }
}

void init_equity(const std::string& leaves_file, const std::string& peg_file) {
  if (!leaves_file.empty())
    HastyEquity::init(leaves_file, peg_file);
  else
    HastyEquity::ensure_initialized(Lexicon::instance().name());
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string mode = "endgames";
    int games = 100;
    uint64_t seed = 1;
    int threads = 1;
    int margin_min = -80;
    int margin_max = 40;
    int margin_step = 1;
    int time_games = 25;
    int incremental = 1;
    int projections = 1;
    scribblez::EndgameSolver::Params params;
    std::string budgets_csv;
    std::string buckets_csv = "20,60";
    std::string leaves_file;
    std::string peg_file;

    po::options_description desc("endgame_bench options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("mode", po::value<std::string>(&mode)->default_value(mode),
                       "endgames | games");
    desc.add_options()(
      "games", po::value<int>(&games)->default_value(games),
      "number of games (capture games in endgames mode; games/config in games mode)");
    desc.add_options()("seed", po::value<uint64_t>(&seed)->default_value(seed),
                       "base seed; game i uses seed+i");
    params.add_options(desc);
    desc.add_options()("budgets", po::value<std::string>(&budgets_csv),
                       "comma-separated budget sweep; overrides --budget when given");
    desc.add_options()("threads,t", po::value<int>(&threads)->default_value(threads),
                       "parallelism (games-mode configs; endgames-mode games)");
    desc.add_options()("margin-min", po::value<int>(&margin_min)->default_value(margin_min),
                       "endgames mode: lowest start-of-endgame margin to sweep");
    desc.add_options()("margin-max", po::value<int>(&margin_max)->default_value(margin_max),
                       "endgames mode: highest start-of-endgame margin to sweep");
    desc.add_options()("margin-step", po::value<int>(&margin_step)->default_value(margin_step),
                       "endgames mode: step between swept margins");
    desc.add_options()("time-games", po::value<int>(&time_games)->default_value(time_games),
                       "endgames mode: how many games to sweep single-threaded; only these are "
                       "timed, and only they feed the cost table");
    desc.add_options()("incremental", po::value<int>(&incremental)->default_value(incremental),
                       "solver incremental move-list maintenance (1 on, 0 off); changes only "
                       "speed, never results");
    desc.add_options()("projections", po::value<int>(&projections)->default_value(projections),
                       "endgames mode: 1 ends a solved endgame at its certificate, as self-play "
                       "generation does; 0 plays it out move by move");
    desc.add_options()("spread-buckets",
                       po::value<std::string>(&buckets_csv)->default_value(buckets_csv),
                       "games mode: ascending |spread| thresholds bucketing head-to-head games by "
                       "their baseline bag-empty spread");
    desc.add_options()("leaves-file", po::value<std::string>(&leaves_file),
                       "path to leaves.klv2 (optional; defaults to the active lexicon's)");
    desc.add_options()("peg-file", po::value<std::string>(&peg_file)->default_value(""),
                       "path to preendgame.json (optional)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    scribblez::init_equity(leaves_file, peg_file);
    const scribblez::Dictionary& dict = scribblez::load_dictionary_or_throw();
    const std::vector<uint64_t> budgets = budgets_csv.empty()
                                            ? std::vector<uint64_t>{params.budget}
                                            : scribblez::parse_budgets(budgets_csv);
    const std::vector<int> thresholds = scribblez::parse_thresholds(buckets_csv);
    if (threads < 1) threads = 1;
    if (margin_step < 1) margin_step = 1;
    if (time_games < 0) time_games = 0;

    if (mode == "endgames") {
      scribblez::run_endgames_mode(dict, seed, games, threads, budgets, params, incremental != 0,
                                   projections != 0, margin_min, margin_max, margin_step,
                                   time_games);
    } else if (mode == "games") {
      scribblez::run_games_mode(dict, seed, games, threads, budgets, params, incremental != 0,
                                thresholds);
    } else {
      std::cerr << "unknown --mode '" << mode << "' (expected endgames or games)\n";
      return 1;
    }
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
