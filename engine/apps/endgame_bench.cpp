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

  MoveDecision make_move(const MoveRequest& req) override {
    const Move m = bot_.make_move(req).move;
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

// One position's solve outcome, tagged with its |spread| bucket so any bucket
// (or the whole batch) can be aggregated after the fact.
struct SolveSample {
  int bucket;
  double us;
  uint64_t nodes;
  int depth;
  bool differ;        // solver's move differs from the captured HastyBot move
  bool class_proven;  // the solve proved the win/draw/loss class
  bool certificate;   // the proof can fast-track the game: it carries a
                      // certificate continuation, or its chosen move itself
                      // ends the game (nothing remains to project)
  bool proven;        // the solve's final value is proven
};

// Solve every captured position once at (budget, objective), tagging each
// sample with its decision-point |spread| bucket.
std::vector<SolveSample> solve_positions(const Dictionary& dict,
                                         const std::vector<CapturedPosition>& positions,
                                         uint64_t budget, int plies, EndgameObjective objective,
                                         bool futility, const std::vector<int>& thresholds) {
  EndgameSolver solver;
  solver.set_outplay_futility(futility);
  std::vector<SolveSample> samples;
  samples.reserve(positions.size());
  for (const CapturedPosition& p : positions) {
    solver.clear();  // independent samples: don't let one solve warm the next
    const auto t0 = std::chrono::steady_clock::now();
    const EndgameResult r = solver.solve(p.board, dict, p.my_rack, p.opp_rack, p.my_score,
                                         p.opp_score, p.scoreless, budget, plies, objective);
    const auto t1 = std::chrono::steady_clock::now();
    const bool class_proven = r.proven_class != EndgameResult::kClassUnknown;
    const bool best_ends_game =
      (r.best.type() == MoveType::PLAY && r.best.num_glyphs() == p.my_rack.size()) ||
      (r.best.type() == MoveType::PASS && p.scoreless >= 1);
    samples.push_back({bucket_of(std::abs(p.my_score - p.opp_score), thresholds),
                       std::chrono::duration<double>(t1 - t0).count() * 1e6, r.nodes,
                       r.depth_completed, !(r.best == p.hasty_move), class_proven,
                       class_proven && (best_ends_game || !r.continuation.empty()), r.proven});
  }
  return samples;
}

// Aggregate over one (budget, spread-bucket) cell of solve samples.
struct SolveStats {
  size_t positions;
  double mean_us;
  double median_us;
  double nodes_per_s;
  double mean_depth;
  double pct_differ;
  double pct_class;
  double pct_cert;
  double pct_proven;
  double pct_budget;  // mean share of the node budget actually spent: the
                      // complement of what proof early exits hand back
};

// Aggregate the samples in bucket `bucket`, or every sample when bucket < 0.
SolveStats aggregate_stats(const std::vector<SolveSample>& samples, int bucket, uint64_t budget) {
  std::vector<double> per_us;
  double total_us = 0.0;
  uint64_t total_nodes = 0;
  long total_depth = 0;
  size_t differ = 0, class_proven = 0, certificates = 0, proven = 0;
  for (const SolveSample& sm : samples) {
    if (bucket >= 0 && sm.bucket != bucket) continue;
    per_us.push_back(sm.us);
    total_us += sm.us;
    total_nodes += sm.nodes;
    total_depth += sm.depth;
    differ += sm.differ ? 1 : 0;
    class_proven += sm.class_proven ? 1 : 0;
    certificates += sm.certificate ? 1 : 0;
    proven += sm.proven ? 1 : 0;
  }
  std::sort(per_us.begin(), per_us.end());
  const size_t n = per_us.size();
  SolveStats st{};
  st.positions = n;
  if (n == 0) return st;
  st.mean_us = total_us / static_cast<double>(n);
  st.median_us = per_us[n / 2];
  st.nodes_per_s = total_us > 0 ? 1e6 * static_cast<double>(total_nodes) / total_us : 0.0;
  st.mean_depth = static_cast<double>(total_depth) / static_cast<double>(n);
  st.pct_differ = 100.0 * static_cast<double>(differ) / static_cast<double>(n);
  st.pct_class = 100.0 * static_cast<double>(class_proven) / static_cast<double>(n);
  st.pct_cert = 100.0 * static_cast<double>(certificates) / static_cast<double>(n);
  st.pct_proven = 100.0 * static_cast<double>(proven) / static_cast<double>(n);
  st.pct_budget = 100.0 * static_cast<double>(total_nodes) /
                  (static_cast<double>(budget) * static_cast<double>(n));
  return st;
}

void print_solve_row(uint64_t budget, const std::string& bucket, const SolveStats& s) {
  std::printf(
    "%10llu %8s %9zu %11.1f %11.1f %13.0f %10.2f %8.1f%% %7.1f%% %7.1f%% %8.1f%% %8.1f%%\n",
    static_cast<unsigned long long>(budget), bucket.c_str(), s.positions, s.mean_us, s.median_us,
    s.nodes_per_s, s.mean_depth, s.pct_differ, s.pct_class, s.pct_cert, s.pct_proven, s.pct_budget);
}

void run_solves_mode(const Dictionary& dict, uint64_t base_seed, int games,
                     const std::vector<uint64_t>& budgets, int plies, EndgameObjective objective,
                     const std::vector<int>& thresholds, bool futility) {
  const std::vector<CapturedPosition> positions = capture_positions(dict, base_seed, games);
  std::printf(
    "solves mode: %d games, %zu bag-empty positions, plies=%d, objective=%s, futility=%d\n\n",
    games, positions.size(), plies, endgame_objective_name(objective), futility ? 1 : 0);
  std::printf("%10s %8s %9s %11s %11s %13s %10s %9s %8s %8s %9s %9s\n", "budget", "|spread|",
              "positions", "mean us", "p50 us", "nodes/s", "mean depth", "%% differ", "%% class",
              "%% cert", "%% proven", "%% budget");
  // Per budget: one solve of every position, then per-bucket rows (small
  // buckets: decision accuracy; large: the break-out imperative) and an "all"
  // row for the single overall number.
  for (uint64_t b : budgets) {
    const std::vector<SolveSample> samples =
      solve_positions(dict, positions, b, plies, objective, futility, thresholds);
    for (int k = 0; k <= static_cast<int>(thresholds.size()); ++k) {
      const SolveStats s = aggregate_stats(samples, k, b);
      if (s.positions == 0) continue;
      print_solve_row(b, bucket_label(k, thresholds), s);
    }
    print_solve_row(b, "all", aggregate_stats(samples, -1, b));
  }
}

using AgentFactory = std::function<std::unique_ptr<Agent>(int thread_id)>;

// Play `games` seeded games (seed base_seed+i) of factory-built agents and
// return the wall-clock seconds. Threads split the game indices into contiguous
// chunks; each thread builds its own pair of agents once and reuses them.
double run_config(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                  const AgentFactory& make0, const AgentFactory& make1, bool fast_track) {
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
        g.set_respect_projections(fast_track);
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
                    const std::vector<int>& thresholds, bool fast_track) {
  std::printf("games mode: %d games/config, threads=%d, plies=%d, objective=%s, fast-track=%d\n\n",
              games, threads, plies, endgame_objective_name(objective), fast_track ? 1 : 0);
  std::printf("%-22s %11s %10s %10s %10s %10s\n", "config", "budget", "total s", "s/game",
              "games/s", "ratio");

  const double base_s = run_config(dict, base_seed, games, threads, hasty_factory(),
                                   hasty_factory(), /*fast_track=*/false);
  print_games_row("hasty-vs-hasty", "-", base_s, games, 1.0);

  // The cost sweep honors --fast-track (a self-play generator's configuration);
  // the head-to-head strength protocol below never does -- fast-tracking would
  // substitute the proof's optimal replies for the live opponent's moves.
  for (uint64_t b : budgets) {
    const AgentFactory f = endgame_factory(b, plies, objective);
    const double s = run_config(dict, base_seed, games, threads, f, f, fast_track);
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
    H2H total;
    for (size_t k = 0; k < buckets.size(); ++k) {
      const H2H& h = buckets[k];
      total.spread_sum += h.spread_sum;
      total.games += h.games;
      total.wins += h.wins;
      total.draws += h.draws;
      total.losses += h.losses;
      if (h.games == 0) continue;
      const std::string label =
        k + 1 == buckets.size() ? "none" : bucket_label(static_cast<int>(k), thresholds);
      std::printf("%11llu %9s %8d %16.2f %8d %8d %8d\n", static_cast<unsigned long long>(b),
                  label.c_str(), h.games, static_cast<double>(h.spread_sum) / h.games, h.wins,
                  h.draws, h.losses);
    }
    std::printf("%11llu %9s %8d %16.2f %8d %8d %8d\n", static_cast<unsigned long long>(b), "all",
                total.games, static_cast<double>(total.spread_sum) / total.games, total.wins,
                total.draws, total.losses);
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
    bool fast_track = false;
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
    desc.add_options()("fast-track", po::bool_switch(&fast_track),
                       "games mode: the cost sweep's games respect agent projections (the "
                       "self-play break-out); head-to-head games never do");
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
    const scribblez::EndgameObjective objective = scribblez::parse_endgame_objective(objective_str);
    if (threads < 1) threads = 1;

    if (mode == "solves") {
      scribblez::run_solves_mode(dict, seed, games, budgets, plies, objective, thresholds,
                                 !no_futility);
    } else if (mode == "games") {
      scribblez::run_games_mode(dict, seed, games, threads, budgets, plies, objective, thresholds,
                                fast_track);
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
