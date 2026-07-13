// endgame_bench: measures the phase-B endgame solver's cost and its effect on
// whole-game throughput, so a later pass can pick a per-turn node budget that
// makes an endgame-vs-endgame game take roughly twice as long as a plain
// HastyBot-vs-HastyBot game.
//
// Two modes:
//   --mode=solves : play N HastyBot-vs-HastyBot games, capture every bag-empty
//                   decision point, then solve each captured position at every
//                   budget and report per-budget solve cost, depth reached, and
//                   how often the solver disagrees with HastyBot.
//   --mode=games  : run N seeded full games for hasty-vs-hasty and for
//                   endgame-vs-endgame at every budget (same seeds across
//                   configs) and report wall-time per config and its ratio to
//                   the hasty-vs-hasty baseline.
//
// Usage:
//   endgame_bench [--mode=solves|games] [--games N] [--seed N]
//                 [--budgets 1000,3000,...] [--plies P] [--threads N]
//                 [--lexicon NAME] [--leaves-file PATH] [--peg-file PATH]

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

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
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
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::stoull(tok));
  }
  return out;
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

// Per-budget aggregate over one solve of every captured position.
struct SolveStats {
  uint64_t budget;
  size_t positions;
  double mean_us;
  double median_us;
  double p95_us;
  double nodes_per_s;
  double mean_depth;
  double pct_differ;
};

SolveStats solve_all(const Dictionary& dict, const std::vector<CapturedPosition>& positions,
                     uint64_t budget, int plies) {
  EndgameSolver solver;
  std::vector<double> per_us;
  per_us.reserve(positions.size());
  double total_s = 0.0;
  uint64_t total_nodes = 0;
  long total_depth = 0;
  size_t differ = 0;
  for (const CapturedPosition& p : positions) {
    solver.clear();  // independent samples: don't let one solve warm the next
    const auto t0 = std::chrono::steady_clock::now();
    const EndgameResult r = solver.solve(p.board, dict, p.my_rack, p.opp_rack, p.my_score,
                                         p.opp_score, p.scoreless, budget, plies);
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    per_us.push_back(s * 1e6);
    total_s += s;
    total_nodes += r.nodes;
    total_depth += r.depth_completed;
    if (!(r.best == p.hasty_move)) ++differ;
  }

  std::sort(per_us.begin(), per_us.end());
  const size_t n = per_us.size();
  SolveStats st;
  st.budget = budget;
  st.positions = n;
  st.mean_us = n ? std::accumulate(per_us.begin(), per_us.end(), 0.0) / n : 0.0;
  st.median_us = n ? per_us[n / 2] : 0.0;
  st.p95_us = n ? per_us[std::min(n - 1, static_cast<size_t>(0.95 * n))] : 0.0;
  st.nodes_per_s = total_s > 0 ? static_cast<double>(total_nodes) / total_s : 0.0;
  st.mean_depth = n ? static_cast<double>(total_depth) / n : 0.0;
  st.pct_differ = n ? 100.0 * static_cast<double>(differ) / n : 0.0;
  return st;
}

void run_solves_mode(const Dictionary& dict, uint64_t base_seed, int games,
                     const std::vector<uint64_t>& budgets, int plies) {
  const std::vector<CapturedPosition> positions = capture_positions(dict, base_seed, games);
  std::printf("solves mode: %d games, %zu bag-empty positions, plies=%d\n\n", games,
              positions.size(), plies);
  std::printf("%10s %9s %11s %11s %11s %13s %10s %9s\n", "budget", "positions", "mean us", "p50 us",
              "p95 us", "nodes/s", "mean depth", "%% differ");
  for (uint64_t b : budgets) {
    const SolveStats s = solve_all(dict, positions, b, plies);
    std::printf("%10llu %9zu %11.1f %11.1f %11.1f %13.0f %10.2f %8.1f%%\n",
                static_cast<unsigned long long>(s.budget), s.positions, s.mean_us, s.median_us,
                s.p95_us, s.nodes_per_s, s.mean_depth, s.pct_differ);
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

AgentFactory endgame_factory(uint64_t budget, int plies) {
  return [budget, plies](int tid) -> std::unique_ptr<Agent> {
    EndgameHastyBotAgent::Params p;
    p.hasty = HastyBotAgent::Params{.thread_id = tid, .name = "EndgameHastyBot"};
    p.endgame_nodes = budget;
    p.endgame_plies = plies;
    return std::make_unique<EndgameHastyBotAgent>(p);
  };
}

// Play `games` seeded games of the endgame bot against a plain HastyBot and
// return the endgame bot's mean final spread (its score minus the opponent's).
// Each seed is played twice with the seats mirrored, and the pair's two spreads
// are averaged: the same tile-draw sequence serves both bots symmetrically, so
// per-seed draw luck (who gets the blanks) cancels instead of dominating the
// variance. A positive value is the head-to-head strength signal: points the
// endgame solver wins over greedy play. Single-threaded so the per-game spreads
// are deterministic in `base_seed`.
double endgame_vs_hasty_spread(const Dictionary& dict, uint64_t base_seed, int games,
                               uint64_t budget, int plies) {
  const AgentFactory eg = endgame_factory(budget, plies);
  const AgentFactory hb = hasty_factory();
  long total_spread = 0;
  int played = 0;
  for (int i = 0; i < (games + 1) / 2; ++i) {
    const uint64_t seed = base_seed + static_cast<uint64_t>(i);
    for (int eg_seat = 0; eg_seat < 2; ++eg_seat) {
      std::unique_ptr<Agent> p0 = eg_seat == 0 ? eg(0) : hb(0);
      std::unique_ptr<Agent> p1 = eg_seat == 0 ? hb(0) : eg(0);
      Game g(*p0, *p1, dict, seed);
      g.play();
      total_spread += g.score(eg_seat) - g.score(1 - eg_seat);
      ++played;
    }
  }
  return played ? static_cast<double>(total_spread) / played : 0.0;
}

void print_games_row(const char* config, const std::string& budget, double total_s, int games,
                     double ratio) {
  std::printf("%-22s %11s %10.3f %10.4f %10.3f %9.2fx\n", config, budget.c_str(), total_s,
              total_s / games, games / total_s, ratio);
}

void run_games_mode(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                    const std::vector<uint64_t>& budgets, int plies) {
  std::printf("games mode: %d games/config, threads=%d, plies=%d\n\n", games, threads, plies);
  std::printf("%-22s %11s %10s %10s %10s %10s\n", "config", "budget", "total s", "s/game",
              "games/s", "ratio");

  const double base_s =
    run_config(dict, base_seed, games, threads, hasty_factory(), hasty_factory());
  print_games_row("hasty-vs-hasty", "-", base_s, games, 1.0);

  for (uint64_t b : budgets) {
    const AgentFactory f = endgame_factory(b, plies);
    const double s = run_config(dict, base_seed, games, threads, f, f);
    print_games_row("endgame-vs-endgame", std::to_string(b), s, games, s / base_s);
  }

  // Strength evidence: the endgame bot's mean spread against a plain HastyBot,
  // alternating seats so any first-move/board advantage averages out. Run
  // single-threaded for deterministic per-game spreads.
  std::printf("\nhead-to-head (endgame-vs-hasty, alternating seats):\n");
  std::printf("%11s %16s\n", "budget", "mean eg spread");
  for (uint64_t b : budgets) {
    const double spread = endgame_vs_hasty_spread(dict, base_seed, games, b, plies);
    std::printf("%11llu %16.2f\n", static_cast<unsigned long long>(b), spread);
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
    std::string budgets_csv = "1000,3000,10000,30000,100000,300000";
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
    desc.add_options()("leaves-file", po::value<std::string>(&leaves_file),
                       "path to leaves.klv2 (optional; defaults to the active lexicon's)");
    desc.add_options()("peg-file", po::value<std::string>(&peg_file)->default_value(""),
                       "path to preendgame.json (optional)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    scribblez::init_equity(leaves_file, peg_file);
    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    const std::vector<uint64_t> budgets = scribblez::parse_budgets(budgets_csv);
    if (threads < 1) threads = 1;

    if (mode == "solves") {
      scribblez::run_solves_mode(dict, seed, games, budgets, plies);
    } else if (mode == "games") {
      scribblez::run_games_mode(dict, seed, games, threads, budgets, plies);
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
