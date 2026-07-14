// endgame_bench: measures the endgame solver's cost and its effect on
// whole-game throughput and strength, per solver objective (see
// EndgameObjective), with results bucketed by bag-empty spread -- small
// buckets measure decision accuracy (the endgame still decides those games),
// large buckets measure the break-out imperative (prove the decided game's
// class fast and hand the budget back).
//
// Two modes:
//   --mode=solves : play N HastyBot-vs-HastyBot games, capture every bag-empty
//                   decision point, then solve each captured position at every
//                   budget and report, per (budget, |spread| bucket): solve
//                   cost, depth, how often the solver disagrees with HastyBot,
//                   class-proof and value-proof rates, and the share of the
//                   node budget actually spent.
//   --mode=games  : run N seeded full games for hasty-vs-hasty and for
//                   endgame-vs-endgame at every budget (same seeds across
//                   configs), report wall-time ratios to the hasty-vs-hasty
//                   baseline, then the seat-mirrored head-to-head record
//                   against plain HastyBot bucketed by each seed's baseline
//                   bag-empty spread.
//
// Usage:
//   endgame_bench [--mode=solves|games] [--games N] [--seed N]
//                 [--budgets 1000,3000,...] [--plies P] [--threads N]
//                 [--objective lexicographic|first-win|spread]
//                 [--spread-buckets 20,60] [--no-futility]
//                 [--lexicon NAME] [--leaves-file PATH] [--peg-file PATH]
//
// Measured results (cost ratios, strength-vs-budget, and methodology) are
// summarized in docs/endgame_bench_results.md.

#include "agent/agent.h"
#include "agent/endgame_hasty_bot.h"
#include "agent/macondo_bot.h"
#include "endgame/endgame_solver.h"
#include "game/board.h"
#include "game/game.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "selfplay/game_runner.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/string.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace scribblez {
namespace {

// A bag-empty decision point drawn from HastyBot self-play: everything the
// solver needs to re-solve it, plus the move HastyBot actually made there (to
// score how often the solver disagrees).
struct CapturedPosition {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score;
  int opp_score;
  int scoreless;
  Move hasty_move;
};

// A HastyBot that records every bag-empty position it is asked to move on. It
// tracks the consecutive-scoreless-turn count from observe_move exactly as the
// game loop does, so each captured position carries the solver's scoreless input.
class CapturingHastyBot : public Agent {
 public:
  CapturingHastyBot(int thread_id, const std::string& name, std::vector<CapturedPosition>& sink)
      : Agent(thread_id, name), bot_({.thread_id = thread_id, .name = name}), sink_(sink) {}

  Move make_move(const MoveRequest& req) override {
    const Move m = bot_.make_move(req);
    if (req.bag_size == 0) {
      sink_.push_back(
        {req.board, req.my_rack, req.opp_rack, req.my_score, req.opp_score, scoreless_, m});
    }
    return m;
  }

  void observe_move(const Move& move) override {
    if (move.type() == MoveType::PLAY)
      scoreless_ = 0;
    else
      ++scoreless_;
  }

  void begin_game() override { scoreless_ = 0; }

 private:
  HastyBotAgent bot_;
  std::vector<CapturedPosition>& sink_;
  int scoreless_ = 0;
};

// Parse a comma-separated budget list ("1000,3000,10000") into node counts.
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
// bucket k holds [t_{k-1}, t_k), with a final unbounded bucket. Small buckets
// are the accuracy regime -- endgame decisions there swing the game result --
// while large buckets are already decided, the regime where the break-out
// objective should prove the class fast and stop.
int bucket_of(int abs_spread, const std::vector<int>& thresholds) {
  int k = 0;
  while (k < static_cast<int>(thresholds.size()) && abs_spread >= thresholds[k]) ++k;
  return k;
}

std::string bucket_label(int k, const std::vector<int>& thresholds) {
  const int lo = k == 0 ? 0 : thresholds[k - 1];
  if (k == static_cast<int>(thresholds.size())) return std::to_string(lo) + "+";
  return std::to_string(lo) + "-" + std::to_string(thresholds[k] - 1);
}

const char* objective_name(EndgameObjective objective) {
  switch (objective) {
    case EndgameObjective::kSpread:
      return "spread";
    case EndgameObjective::kFirstWin:
      return "first-win";
    case EndgameObjective::kLexicographic:
      return "lexicographic";
  }
  return "?";
}

// Play `games` HastyBot-vs-HastyBot games seeded base_seed+i and return every
// bag-empty position they passed through.
std::vector<CapturedPosition> capture_positions(const Dictionary& dict, uint64_t base_seed,
                                                int games) {
  std::vector<CapturedPosition> positions;
  for (int i = 0; i < games; ++i) {
    CapturingHastyBot a0(0, "A", positions), a1(0, "B", positions);
    Game g(a0, a1, dict, base_seed + static_cast<uint64_t>(i));
    g.play();
  }
  return positions;
}

// Aggregate over one solve of each position in one (budget, spread-bucket)
// cell.
struct SolveStats {
  size_t positions;
  double mean_us;
  double median_us;
  double nodes_per_s;
  double mean_depth;
  double pct_differ;  // solver's move differs from the captured HastyBot move
  double pct_class;   // solves that proved the win/draw/loss class
  double pct_proven;  // solves whose final value is proven
  double pct_budget;  // mean share of the node budget actually spent: the
                      // complement of what proof early exits hand back
};

SolveStats solve_all(const Dictionary& dict, const std::vector<const CapturedPosition*>& positions,
                     uint64_t budget, int plies, EndgameObjective objective, bool futility) {
  EndgameSolver solver;
  solver.set_outplay_futility(futility);
  std::vector<double> per_us;
  per_us.reserve(positions.size());
  double total_s = 0.0;
  uint64_t total_nodes = 0;
  long total_depth = 0;
  double budget_share = 0.0;
  size_t differ = 0;
  size_t class_proven = 0;
  size_t proven = 0;
  for (const CapturedPosition* p : positions) {
    solver.clear();  // independent samples: don't let one solve warm the next
    const auto t0 = std::chrono::steady_clock::now();
    const EndgameResult r = solver.solve(p->board, dict, p->my_rack, p->opp_rack, p->my_score,
                                         p->opp_score, p->scoreless, budget, plies, objective);
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    per_us.push_back(s * 1e6);
    total_s += s;
    total_nodes += r.nodes;
    total_depth += r.depth_completed;
    budget_share += static_cast<double>(r.nodes) / static_cast<double>(budget);
    if (!(r.best == p->hasty_move)) ++differ;
    if (r.proven_class != EndgameResult::kClassUnknown) ++class_proven;
    if (r.proven) ++proven;
  }

  std::sort(per_us.begin(), per_us.end());
  const size_t n = per_us.size();
  SolveStats st;
  st.positions = n;
  st.mean_us = n ? std::accumulate(per_us.begin(), per_us.end(), 0.0) / n : 0.0;
  st.median_us = n ? per_us[n / 2] : 0.0;
  st.nodes_per_s = total_s > 0 ? static_cast<double>(total_nodes) / total_s : 0.0;
  st.mean_depth = n ? static_cast<double>(total_depth) / n : 0.0;
  st.pct_differ = n ? 100.0 * static_cast<double>(differ) / n : 0.0;
  st.pct_class = n ? 100.0 * static_cast<double>(class_proven) / n : 0.0;
  st.pct_proven = n ? 100.0 * static_cast<double>(proven) / n : 0.0;
  st.pct_budget = n ? 100.0 * budget_share / static_cast<double>(n) : 0.0;
  return st;
}

void run_solves_mode(const Dictionary& dict, uint64_t base_seed, int games,
                     const std::vector<uint64_t>& budgets, int plies, EndgameObjective objective,
                     const std::vector<int>& thresholds, bool futility) {
  const std::vector<CapturedPosition> positions = capture_positions(dict, base_seed, games);
  // Bucket the positions by their absolute score spread at the decision point:
  // the small buckets measure decision accuracy, the large ones the break-out
  // imperative (prove the decided game's class and hand the budget back).
  std::vector<std::vector<const CapturedPosition*>> buckets(thresholds.size() + 1);
  for (const CapturedPosition& p : positions)
    buckets[bucket_of(std::abs(p.my_score - p.opp_score), thresholds)].push_back(&p);

  std::printf(
    "solves mode: %d games, %zu bag-empty positions, plies=%d, objective=%s, futility=%d\n\n",
    games, positions.size(), plies, objective_name(objective), futility ? 1 : 0);
  std::printf("%10s %8s %9s %11s %11s %13s %10s %9s %8s %9s %9s\n", "budget", "|spread|",
              "positions", "mean us", "p50 us", "nodes/s", "mean depth", "%% differ", "%% class",
              "%% proven", "%% budget");
  for (uint64_t b : budgets) {
    for (int k = 0; k < static_cast<int>(buckets.size()); ++k) {
      if (buckets[k].empty()) continue;
      const SolveStats s = solve_all(dict, buckets[k], b, plies, objective, futility);
      std::printf("%10llu %8s %9zu %11.1f %11.1f %13.0f %10.2f %8.1f%% %7.1f%% %8.1f%% %8.1f%%\n",
                  static_cast<unsigned long long>(b), bucket_label(k, thresholds).c_str(),
                  s.positions, s.mean_us, s.median_us, s.nodes_per_s, s.mean_depth, s.pct_differ,
                  s.pct_class, s.pct_proven, s.pct_budget);
    }
  }
}

using AgentFactory = std::function<std::unique_ptr<Agent>(int thread_id)>;

// Play `games` seeded games (seed base_seed+i) of factory-built agents and
// return the wall-clock seconds. Threads split the game indices into contiguous
// chunks; each thread builds its own pair of agents once and reuses them.
double run_config(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                  const AgentFactory& make0, const AgentFactory& make1) {
  const auto t0 = std::chrono::steady_clock::now();
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
        g.play();
      }
    });
  }
  for (std::thread& th : pool) th.join();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

AgentFactory hasty_factory() {
  return [](int tid) -> std::unique_ptr<Agent> {
    return std::make_unique<HastyBotAgent>(
      HastyBotAgent::Params{.thread_id = tid, .name = "HastyBot"});
  };
}

AgentFactory endgame_factory(uint64_t budget, int plies, EndgameObjective objective) {
  return [budget, plies, objective](int tid) -> std::unique_ptr<Agent> {
    EndgameHastyBotAgent::Params p;
    p.hasty = HastyBotAgent::Params{.thread_id = tid, .name = "EndgameHastyBot"};
    p.endgame_nodes = budget;
    p.endgame_plies = plies;
    p.endgame_objective = objective;
    return std::make_unique<EndgameHastyBotAgent>(p);
  };
}

// Head-to-head aggregate for one bag-empty-spread bucket: the endgame bot's
// summed final spread plus its win/draw/loss record. Under the first-win
// objective the spread shrinks by construction (winning positions stop
// maximizing it); the W/D/L record is the metric that must hold up.
struct H2H {
  long spread_sum = 0;
  int games = 0;
  int wins = 0;
  int draws = 0;
  int losses = 0;
};

// The absolute score spread at the baseline game's first bag-empty decision
// point, or -1 when the seed's HastyBot-vs-HastyBot game never empties the
// bag. The baseline game is seed-deterministic and agent-independent, so it
// gives every objective and budget the same seat-independent conditioning
// variable for the head-to-head buckets.
int baseline_bag_empty_spread(const Dictionary& dict, uint64_t seed) {
  std::vector<CapturedPosition> caps;
  CapturingHastyBot a0(0, "A", caps), a1(0, "B", caps);
  Game g(a0, a1, dict, seed);
  g.play();
  if (caps.empty()) return -1;
  return std::abs(caps.front().my_score - caps.front().opp_score);
}

// Play `games` seeded games of the endgame bot against a plain HastyBot,
// bucketed by the seed's baseline bag-empty spread (the last bucket holds
// seeds whose baseline game never empties the bag). Each seed is played twice
// with the seats mirrored, so per-seed tile-draw luck (who gets the blanks)
// cancels instead of dominating the variance; unpaired spread estimates are
// not usable. Single-threaded so the per-game results are deterministic in
// `base_seed`.
std::vector<H2H> endgame_vs_hasty(const Dictionary& dict, uint64_t base_seed, int games,
                                  uint64_t budget, int plies, EndgameObjective objective,
                                  const std::vector<int>& thresholds) {
  const AgentFactory eg = endgame_factory(budget, plies, objective);
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
      g.play();
      const int spread = g.score(eg_seat) - g.score(1 - eg_seat);
      h.spread_sum += spread;
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

void run_games_mode(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                    const std::vector<uint64_t>& budgets, int plies, EndgameObjective objective,
                    const std::vector<int>& thresholds) {
  std::printf("games mode: %d games/config, threads=%d, plies=%d, objective=%s\n\n", games, threads,
              plies, objective_name(objective));
  std::printf("%-22s %11s %10s %10s %10s %10s\n", "config", "budget", "total s", "s/game",
              "games/s", "ratio");

  const double base_s =
    run_config(dict, base_seed, games, threads, hasty_factory(), hasty_factory());
  print_games_row("hasty-vs-hasty", "-", base_s, games, 1.0);

  for (uint64_t b : budgets) {
    const AgentFactory f = endgame_factory(b, plies, objective);
    const double s = run_config(dict, base_seed, games, threads, f, f);
    print_games_row("endgame-vs-endgame", std::to_string(b), s, games, s / base_s);
  }

  // Strength evidence: the endgame bot's mean spread and W/D/L record against
  // a plain HastyBot, seats mirrored per seed and bucketed by the seed's
  // baseline bag-empty spread -- accuracy shows up in the small buckets, where
  // the endgame still decides the game. Run single-threaded so the per-game
  // results are deterministic in base_seed.
  std::printf("\nhead-to-head (endgame-vs-hasty, mirrored seats, by baseline bag-empty spread):\n");
  std::printf("%11s %9s %8s %16s %8s %8s %8s\n", "budget", "|spread|", "games", "mean eg spread",
              "W", "D", "L");
  for (uint64_t b : budgets) {
    const std::vector<H2H> buckets =
      endgame_vs_hasty(dict, base_seed, games, b, plies, objective, thresholds);
    for (size_t k = 0; k < buckets.size(); ++k) {
      const H2H& h = buckets[k];
      if (h.games == 0) continue;
      const std::string label =
        k + 1 == buckets.size() ? "none" : bucket_label(static_cast<int>(k), thresholds);
      std::printf("%11llu %9s %8d %16.2f %8d %8d %8d\n", static_cast<unsigned long long>(b),
                  label.c_str(), h.games, static_cast<double>(h.spread_sum) / h.games, h.wins,
                  h.draws, h.losses);
    }
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
    std::string mode = "solves";
    int games = 50;
    uint64_t seed = 1;
    int plies = 25;
    int threads = 1;
    std::string objective_str = "lexicographic";
    bool no_futility = false;
    std::string budgets_csv = "1000,3000,10000,30000,100000,300000";
    std::string buckets_csv = "20,60";
    std::string leaves_file;
    std::string peg_file;

    po::options_description desc("endgame_bench options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("mode", po::value<std::string>(&mode)->default_value(mode),
                       "solves | games");
    desc.add_options()(
      "games", po::value<int>(&games)->default_value(games),
      "number of games (capture games in solves mode; games/config in games mode)");
    desc.add_options()("seed", po::value<uint64_t>(&seed)->default_value(seed),
                       "base seed; game i uses seed+i");
    desc.add_options()("budgets", po::value<std::string>(&budgets_csv)->default_value(budgets_csv),
                       "comma-separated per-turn solver node budgets");
    desc.add_options()("plies", po::value<int>(&plies)->default_value(plies),
                       "solver iterative-deepening depth cap");
    desc.add_options()("threads", po::value<int>(&threads)->default_value(threads),
                       "games-mode parallelism (per-thread agents)");
    desc.add_options()("objective",
                       po::value<std::string>(&objective_str)->default_value(objective_str),
                       "solver objective: lexicographic | first-win | spread");
    desc.add_options()(
      "spread-buckets", po::value<std::string>(&buckets_csv)->default_value(buckets_csv),
      "ascending |spread| thresholds bucketing bag-empty positions (solves mode) and "
      "head-to-head games by their baseline bag-empty spread (games mode)");
    desc.add_options()("no-futility", po::bool_switch(&no_futility),
                       "solves mode: disable outplay-futility pruning, to A/B its effect");
    desc.add_options()("leaves-file", po::value<std::string>(&leaves_file),
                       "path to leaves.klv2 (optional; defaults to the active lexicon's)");
    desc.add_options()("peg-file", po::value<std::string>(&peg_file)->default_value(""),
                       "path to preendgame.json (optional)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    scribblez::init_equity(leaves_file, peg_file);
    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    const std::vector<uint64_t> budgets = scribblez::parse_budgets(budgets_csv);
    const std::vector<int> thresholds = scribblez::parse_thresholds(buckets_csv);
    const scribblez::EndgameObjective objective =
      scribblez::EndgameHastyBotAgent::parse_objective(objective_str);
    if (threads < 1) threads = 1;

    if (mode == "solves") {
      scribblez::run_solves_mode(dict, seed, games, budgets, plies, objective, thresholds,
                                 !no_futility);
    } else if (mode == "games") {
      scribblez::run_games_mode(dict, seed, games, threads, budgets, plies, objective, thresholds);
    } else {
      std::cerr << "unknown --mode '" << mode << "' (expected solves or games)\n";
      return 1;
    }
    return 0;
  } catch (const scribblez::CleanExit&) {
    return 0;
  } catch (const scribblez::Exception&) {
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
    return 1;
  }
}
