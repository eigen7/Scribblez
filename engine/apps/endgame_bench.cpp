// endgame_bench: measures what the endgame solver costs and what it buys.
//
// Two modes:
//   --mode=endgames : play N HastyBot-vs-HastyBot games and capture each one's
//                     first bag-empty position. Then, for every captured
//                     position, synthetically sweep the score margin at the
//                     start of the endgame over [--margin-min, --margin-max]
//                     (from the point of view of the first player to act once
//                     the bag is empty), and for every (margin, budget) report
//                     two tables:
//                       skill -- mean over games of the solver seat's game-value
//                                minus a plain-HastyBot baseline's, in win%
//                                points (a win is 1, a draw 0.5, a loss 0);
//                       cost  -- what solving the endgame does to self-play
//                                throughput: the solver seat's measured endgame
//                                time over the timed games, as a multiple of a
//                                hasty-vs-hasty game (see run_endgames_mode).
//                     Absolute score level is irrelevant -- both agent types
//                     decide off the spread alone -- so each margin sets the
//                     first actor's scores to (margin, 0).
//   --mode=games    : run N seeded full games for hasty-vs-hasty and for
//                     endgame-vs-endgame at every budget (same seeds across
//                     configs) and report wall-time ratios to the
//                     hasty-vs-hasty baseline, then the seat-mirrored
//                     head-to-head win% and W/D/L record against plain HastyBot
//                     bucketed by each seed's baseline bag-empty spread. Every
//                     game respects projections, as self-play generation does.
//
// Usage:
//   endgame_bench [--mode=endgames|games] [--games N] [--seed N]
//                 [--budget N | --budgets 100,220,...] [--plies P]
//                 [--spread-matters 0|1] [--threads N]
//                 [--margin-min M] [--margin-max M] [--margin-step S]
//                 [--time-games N] [--projections 0|1]       (endgames mode)
//                 [--spread-buckets 20,60]                   (games mode)
//                 [--lexicon NAME] [--leaves-file PATH] [--peg-file PATH]
//
// Measured results (cost ratios, strength-vs-budget, and methodology) are
// summarized in docs/endgame_bench_results.md.

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

// Run items [0, items) across `threads` workers, each pulling the next index off
// a shared counter, and pass each worker its own index so it can key per-thread
// resources (the pooled EndgameSolver) off it. Work is handed out dynamically
// because a game's sweep cost varies by orders of magnitude with its position:
// static chunks leave most workers waiting on the slowest chunk.
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

// The first bag-empty decision point of one HastyBot self-play game --
// everything needed to replay the endgame with either agent type. The first
// actor once the bag empties holds my_rack; opp_rack is the reply rack.
struct CapturedEndgame {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless = 0;
};

// A HastyBot that records the game's first bag-empty position it is asked to
// move on. It tracks the consecutive-scoreless-turn count from observe_move
// exactly as the game loop does, so each captured position carries the solver's
// scoreless input.
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

// Parse a comma-separated budget list ("100,220,1600") into node counts.
std::vector<uint64_t> parse_budgets(const std::string& csv) {
  std::vector<uint64_t> out;
  for (const std::string& tok : util::split(csv, ',')) out.push_back(std::stoull(tok));
  return out;
}

// Parse a comma-separated ascending threshold list ("20,60") into spreads.
std::vector<int> parse_thresholds(const std::string& csv) {
  std::vector<int> out;
  for (const std::string& tok : util::split(csv, ',')) out.push_back(std::stoi(tok));
  return out;
}

// Bucket index of an absolute bag-empty spread under ascending `thresholds`:
// bucket k holds [t_{k-1}, t_k), with a final unbounded bucket.
int bucket_of(int abs_spread, const std::vector<int>& thresholds) {
  int k = 0;
  while (k < static_cast<int>(thresholds.size()) && abs_spread >= thresholds[k]) ++k;
  return k;
}

std::string bucket_label(int k, const std::vector<int>& thresholds) {
  const int lo = k == 0 ? 0 : thresholds[k - 1];
  if (k == static_cast<int>(thresholds.size())) return std::format("{}+", lo);
  return std::format("{}-{}", lo, thresholds[k] - 1);
}

// Play `games` HastyBot-vs-HastyBot games seeded base_seed+i and return the
// first bag-empty position of each game that reached one, in seed order. Each
// game is seeded independently, so which worker plays it does not show up in
// the captured position.
std::vector<CapturedEndgame> capture_endgames(const Dictionary& dict, uint64_t base_seed, int games,
                                              int threads) {
  std::vector<CapturedEndgame> caps(games);
  std::vector<char> captured(games, 0);
  parallel_for(games, threads, [&](int worker, int i) {
    bool got = false;
    FirstEndgameCapturer a0(worker, "A", caps[i], got);
    FirstEndgameCapturer a1(worker, "B", caps[i], got);
    Game g(a0, a1, dict, base_seed + static_cast<uint64_t>(i));
    g.play();
    captured[i] = got ? 1 : 0;
  });
  std::vector<CapturedEndgame> out;
  for (int i = 0; i < games; ++i) {
    if (captured[i]) out.push_back(caps[i]);
  }
  return out;
}

// The bag of tiles unseen from the captured endgame: empty, since the board and
// both racks account for the full distribution. play_from then draws nothing,
// so both racks stay exactly as captured.
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

// Mean wall time of a plain HastyBot-vs-HastyBot game, over the same seeds the
// timed sweep games use. This is the denominator the cost table is quoted in:
// what one self-play game costs when nothing solves its endgame.
double hasty_game_ms(const Dictionary& dict, uint64_t base_seed, int games) {
  HastyBotAgent a0(HastyBotAgent::Params{.thread_id = 0, .name = "A"});
  HastyBotAgent a1(HastyBotAgent::Params{.thread_id = 0, .name = "B"});
  const auto t0 = Clock::now();
  for (int i = 0; i < games; ++i) {
    Game g(a0, a1, dict, base_seed + static_cast<uint64_t>(i));
    g.set_respect_projections(true);
    g.play();
  }
  return 1000.0 * seconds_since(t0) / games;
}

// Game value of a finished endgame from the first actor's seat: a win is 1, a
// draw 0.5, a loss 0.
double win_fraction(int spread) {
  if (spread > 0) return 1.0;
  if (spread < 0) return 0.0;
  return 0.5;
}

// --- Margin sweep -----------------------------------------------------------

// One solver-seat endgame playout's result: the first actor's final spread,
// which is deterministic, and the wall time the solver spent on it, which is
// meaningful only for a playout from the single-threaded timed phase.
struct SolverOutcome {
  int spread = 0;
  uint64_t solve_ns = 0;
};

// A solver-seat playout's outcome together with the solver's full totals (the
// caller needs max_solve_nodes for the budget-nesting skip).
struct SolverPlayout {
  int spread = 0;
  EndgameTurnPolicy::SolveTotals totals;
};

// Play one captured endgame with an EndgameHastyBot on the first-actor seat
// (scores set to {margin, 0}) and a plain HastyBot on the reply seat,
// projections respected, and return the first actor's final spread and the
// solver's totals. Agents are rebuilt per call with a fixed thread_id/name so
// greedy HastyBot's deterministic tie-breaks make the spread a pure function of
// the position, margin, and budget (threading cannot perturb it).
// `incremental` toggles the solver's incremental move-list maintenance, which
// changes speed but no result. `projections` toggles whether a proven class
// ends the game at its certificate, as self-play generation has it, or the
// endgame is played out move by move.
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

// The change in the first actor's spread over a plain HastyBot-vs-HastyBot
// playout of the captured endgame (scores start at 0). HastyBot's moves never
// read scores, so this one delta fixes the baseline at every margin: the
// baseline final spread at margin m is m + delta, and the baseline win fraction
// is win_fraction(m + delta).
int baseline_delta(const Dictionary& dict, const CapturedEndgame& cap, int thread_id) {
  HastyBotAgent a0(HastyBotAgent::Params{.thread_id = thread_id, .name = "A"});
  HastyBotAgent a1(HastyBotAgent::Params{.thread_id = thread_id, .name = "B"});
  const Bag pool = empty_pool(cap);
  Game g(a0, a1, dict, /*seed=*/1);
  g.play_from(cap.board, {0, 0}, {cap.my_rack, cap.opp_rack}, pool, /*to_move=*/0);
  return g.score(0) - g.score(1);
}

// The ascending margins the sweep covers: min, min+step, ..., up to max.
std::vector<int> margin_axis(int margin_min, int margin_max, int margin_step) {
  std::vector<int> margins;
  for (int m = margin_min; m <= margin_max; m += margin_step) margins.push_back(m);
  return margins;
}

// Solve one (game, margin) column of the grid -- every budget at that margin --
// into `grid[cell_base + budget index]`.
//
// Budgets are processed in DESCENDING order so the budget-nesting skip can
// reuse a larger budget's bit-identical result: once a run's max_solve_nodes is
// <= a smaller budget b', re-running at b' changes nothing (no solve hit the
// larger cap, and any solve declined for having more root moves than the larger
// budget stays declined at b'), so its measured time stands for the smaller
// budget's too. The skip is unsound when spread_matters (its half-budget class
// pass makes behavior depend on the budget value itself), so it is disabled
// there. `skipped` counts the cells the skip filled without a playout.
//
// A column, rather than a whole game, is the sweep's unit of parallel work:
// per-position solve cost spans orders of magnitude, so whole-game items leave
// every worker waiting on the slowest game.
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

// Print the two margin-sweep tables from the filled grid: skill (solver win%
// minus baseline win%, plus a trailing baseline win% column) over every game,
// and cost over the first `timed_games`, margins as ascending rows and budgets
// as columns. Cost is quoted as what a self-play game costs with the endgame
// solved, relative to `baseline_ms` -- the same currency as the games-mode
// throughput ratios, and the one that says what the solver does to generation
// throughput. 1.00x is free.
void print_sweep_tables(const std::vector<int>& margins, const std::vector<uint64_t>& budgets,
                        const std::vector<int>& d0, const std::vector<SolverOutcome>& grid,
                        int timed_games, double baseline_ms) {
  const int ms = static_cast<int>(margins.size());
  const int bs = static_cast<int>(budgets.size());
  const int games = static_cast<int>(d0.size());

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
        const SolverOutcome& o = grid[(static_cast<size_t>(g) * ms + mi) * bs + bi];
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
        ns += static_cast<double>(grid[(static_cast<size_t>(g) * ms + mi) * bs + bi].solve_ns);
      }
      std::printf(" %10.3f", 1.0 + ns / timed_games / 1e6 / baseline_ms);
    }
    std::printf("\n");
  }
}

// Sweep every captured endgame over the margin x budget grid. The first
// `time_games` games run alone on one thread and are the only ones the cost
// table reads: a solve's wall time is only worth reporting when nothing else on
// the machine is competing for cores, caches, and clock. The rest run across
// `threads` workers and contribute their (deterministic) spreads to the skill
// table, which is where sample size buys accuracy.
void run_endgames_mode(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                       const std::vector<uint64_t>& budgets, EndgameSolver::Params params,
                       bool incremental, bool projections, int margin_min, int margin_max,
                       int margin_step, int time_games) {
  const std::vector<CapturedEndgame> caps = capture_endgames(dict, base_seed, games, threads);
  const int g_count = static_cast<int>(caps.size());
  const std::vector<int> margins = margin_axis(margin_min, margin_max, margin_step);
  const int ms = static_cast<int>(margins.size());
  const int bs = static_cast<int>(budgets.size());
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

  std::vector<SolverOutcome> grid(static_cast<size_t>(g_count) * ms * bs);
  std::atomic<uint64_t> skipped{0};
  const auto sweep = [&](int worker, int g, int mi) {
    sweep_column(dict, caps[g], margins[mi], desc_budgets, params, incremental, projections, worker,
                 (static_cast<size_t>(g) * ms + mi) * bs, grid, skipped);
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

  const uint64_t total_cells = static_cast<uint64_t>(g_count) * ms * bs;
  std::printf("budget-nesting skip: %llu of %llu solver playouts avoided\n",
              static_cast<unsigned long long>(skipped.load()),
              static_cast<unsigned long long>(total_cells));

  print_sweep_tables(margins, budgets, d0, grid, timed, baseline_ms);
}

// --- Games mode -------------------------------------------------------------

using AgentFactory = std::function<std::unique_ptr<Agent>(int thread_id)>;

// Play `games` seeded games (seed base_seed+i) of factory-built agents and
// return the wall-clock seconds. Threads split the game indices into contiguous
// chunks; each thread builds its own pair of agents once and reuses them. All
// games respect agent projections, as self-play does.
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
        Game g(*p0, *p1, dict, base_seed + static_cast<uint64_t>(i));
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

// Head-to-head record for one bag-empty-spread bucket: the endgame bot's
// win/draw/loss counts against a plain HastyBot.
struct H2H {
  int games = 0;
  int wins = 0;
  int draws = 0;
  int losses = 0;
};

// The absolute score spread at the baseline game's first bag-empty decision
// point, or -1 when the seed's HastyBot-vs-HastyBot game never empties the
// bag. The baseline game is seed-deterministic and agent-independent, so it
// gives every configuration the same seat-independent conditioning variable
// for the head-to-head buckets.
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

// Play `games` seeded games of the endgame bot against a plain HastyBot,
// bucketed by the seed's baseline bag-empty spread (the last bucket holds
// seeds whose baseline game never empties the bag). Each seed is played twice
// with the seats mirrored, so per-seed tile-draw luck (who gets the blanks)
// cancels instead of dominating the variance; unpaired spread estimates are
// not usable. Games respect projections, so once the bot proves a class the
// recorded spread is the certificate line's -- the production semantics.
// Single-threaded so the per-game results are deterministic in `base_seed`.
std::vector<H2H> endgame_vs_hasty(const Dictionary& dict, uint64_t base_seed, int games,
                                  const EndgameSolver::Params& params, bool incremental,
                                  const std::vector<int>& thresholds) {
  const AgentFactory eg = endgame_factory(params, incremental);
  const AgentFactory hb = hasty_factory();
  std::vector<H2H> buckets(thresholds.size() + 2);
  for (int i = 0; i < (games + 1) / 2; ++i) {
    const uint64_t seed = base_seed + static_cast<uint64_t>(i);
    const int be = baseline_bag_empty_spread(dict, seed);
    H2H& h = buckets[be < 0 ? buckets.size() - 1 : static_cast<size_t>(bucket_of(be, thresholds))];
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

// One head-to-head row: the endgame bot's win% ((W + 0.5*D)/games) and its
// W/D/L counts for a bucket.
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

  // Strength evidence: the endgame bot's win% and W/D/L record against a plain
  // HastyBot, seats mirrored per seed and bucketed by the seed's baseline
  // bag-empty spread -- accuracy shows up in the small buckets, where the
  // endgame still decides the game. Run single-threaded so the per-game results
  // are deterministic in base_seed.
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
      const std::string label =
        k + 1 == buckets.size() ? "none" : bucket_label(static_cast<int>(k), thresholds);
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
