// endgame_bench: measures what the endgame solver costs and what it buys,
// with results bucketed by bag-empty spread -- small buckets measure decision
// accuracy (the endgame still decides those games), large buckets are already
// decided, the regime the break-out machinery exists for.
//
// Two modes:
//   --mode=endgames : play N HastyBot-vs-HastyBot games, timing each whole
//                     game (A) and its endgame phase (B, from the first
//                     bag-empty decision to the end). Then play each captured
//                     endgame out with solver agents on both seats,
//                     projections respected, timing the whole endgame (E).
//                     Reports, per (budget, |spread| bucket): the endgame-
//                     phase multiplier E/B and the whole-game multiplier
//                     (A-B+E)/A the solver imposes on self-play.
//   --mode=games    : run N seeded full games for hasty-vs-hasty and for
//                     endgame-vs-endgame at every budget (same seeds across
//                     configs; endgame games respect projections, as self-play
//                     generation does) and report wall-time ratios to the
//                     hasty-vs-hasty baseline, then the seat-mirrored
//                     head-to-head record against plain HastyBot bucketed by
//                     each seed's baseline bag-empty spread. Head-to-head
//                     games never fast-track: substituting the proof's replies
//                     for the live opponent's moves would corrupt the paired
//                     protocol.
//
// Usage:
//   endgame_bench [--mode=endgames|games] [--games N] [--seed N]
//                 [--budget N | --budgets 100,220,...] [--plies P]
//                 [--spread-matters 0|1] [--spread-buckets 20,60]
//                 [--lexicon NAME] [--leaves-file PATH] [--peg-file PATH]
//
// Measured results (cost ratios, strength-vs-budget, and methodology) are
// summarized in docs/endgame_bench_results.md.

#include "agent/agent.h"
#include "agent/endgame_hasty_bot.h"
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
#include <string>
#include <thread>
#include <vector>

namespace scribblez {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// The first bag-empty decision point of one HastyBot self-play game --
// everything needed to replay the endgame with solver agents -- plus the
// baseline game's timings around it.
struct CapturedEndgame {
  Board board;
  Rack my_rack;
  Rack opp_rack;
  int my_score = 0;
  int opp_score = 0;
  int scoreless = 0;
  double game_s = 0.0;     // whole hasty-vs-hasty game (A)
  double endgame_s = 0.0;  // its endgame phase: first bag-empty decision to end (B)
};

// A HastyBot that records the game's first bag-empty position it is asked to
// move on, with the moment it was asked (the endgame phase's start). It tracks
// the consecutive-scoreless-turn count from observe_move exactly as the game
// loop does.
class FirstEndgameCapturer : public Agent {
 public:
  FirstEndgameCapturer(int thread_id, const std::string& name, CapturedEndgame& sink,
                       bool& captured, Clock::time_point& endgame_start)
      : Agent(thread_id, name),
        bot_({.thread_id = thread_id, .name = name}),
        sink_(sink),
        captured_(captured),
        endgame_start_(endgame_start) {}

  MoveDecision make_move(const MoveRequest& req) override {
    if (req.bag_size == 0 && !captured_) {
      captured_ = true;
      endgame_start_ = Clock::now();
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

  void begin_game() override { scoreless_ = 0; }

 private:
  HastyBotAgent bot_;
  CapturedEndgame& sink_;
  bool& captured_;
  Clock::time_point& endgame_start_;
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
  if (k == static_cast<int>(thresholds.size())) return std::to_string(lo) + "+";
  return std::to_string(lo) + "-" + std::to_string(thresholds[k] - 1);
}

// Play `games` HastyBot-vs-HastyBot games seeded base_seed+i and return each
// game that reached an endgame, with its baseline timings.
std::vector<CapturedEndgame> capture_endgames(const Dictionary& dict, uint64_t base_seed,
                                              int games) {
  std::vector<CapturedEndgame> out;
  for (int i = 0; i < games; ++i) {
    CapturedEndgame cap;
    bool captured = false;
    Clock::time_point endgame_start;
    FirstEndgameCapturer a0(0, "A", cap, captured, endgame_start);
    FirstEndgameCapturer a1(0, "B", cap, captured, endgame_start);
    const auto t0 = Clock::now();
    Game g(a0, a1, dict, base_seed + static_cast<uint64_t>(i));
    g.play();
    const auto t1 = Clock::now();
    cap.game_s = std::chrono::duration<double>(t1 - t0).count();
    if (!captured) continue;  // the game never emptied the bag
    cap.endgame_s = std::chrono::duration<double>(t1 - endgame_start).count();
    out.push_back(cap);
  }
  return out;
}

// The bag of tiles unseen from the captured endgame: empty, since the board
// and both racks account for the full distribution.
Bag empty_pool(const CapturedEndgame& cap, uint64_t seed) {
  Bag pool(seed);
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

// One (budget, bucket) cell of endgame playouts: baseline and solver timings.
struct EndgameTimings {
  int games = 0;
  double game_s = 0.0;     // sum of baseline whole-game times (A)
  double endgame_s = 0.0;  // sum of baseline endgame-phase times (B)
  double solver_s = 0.0;   // sum of solver-agent endgame playout times (E)
};

// Play each captured endgame to completion with solver agents on both seats,
// projections respected (the self-play configuration), and accumulate the
// timings per bucket. The last bucket aggregates everything.
std::vector<EndgameTimings> play_endgames(const Dictionary& dict,
                                          const std::vector<CapturedEndgame>& captured,
                                          const EndgameSolver::Params& params,
                                          const std::vector<int>& thresholds) {
  EndgameHastyBotAgent::Params agent_params;
  agent_params.hasty = HastyBotAgent::Params{.thread_id = 0, .name = "EndgameHastyBot"};
  agent_params.solver = params;
  EndgameHastyBotAgent a0(agent_params), a1(agent_params);

  std::vector<EndgameTimings> buckets(thresholds.size() + 2);
  for (const CapturedEndgame& cap : captured) {
    const int k = bucket_of(std::abs(cap.my_score - cap.opp_score), thresholds);
    const Bag pool = empty_pool(cap, /*seed=*/1);
    Game g(a0, a1, dict, /*seed=*/1);  // the empty pool leaves nothing to draw
    g.set_respect_projections(true);
    const auto t0 = Clock::now();
    g.play_from(cap.board, {cap.my_score, cap.opp_score}, {cap.my_rack, cap.opp_rack}, pool,
                /*to_move=*/0);
    const double e = seconds_since(t0);
    for (EndgameTimings* cell : {&buckets[k], &buckets.back()}) {
      ++cell->games;
      cell->game_s += cap.game_s;
      cell->endgame_s += cap.endgame_s;
      cell->solver_s += e;
    }
  }
  return buckets;
}

void run_endgames_mode(const Dictionary& dict, uint64_t base_seed, int games,
                       const std::vector<uint64_t>& budgets, EndgameSolver::Params params,
                       const std::vector<int>& thresholds) {
  const std::vector<CapturedEndgame> captured = capture_endgames(dict, base_seed, games);
  std::printf("endgames mode: %d games, %zu reached an endgame, plies=%d, spread-matters=%d\n\n",
              games, captured.size(), params.plies, params.spread_matters ? 1 : 0);
  std::printf("%10s %8s %7s %12s %15s %14s %11s %9s\n", "budget", "|spread|", "games",
              "game ms (A)", "endgame ms (B)", "solver ms (E)", "endgame x", "game x");
  for (uint64_t b : budgets) {
    params.budget = b;
    const std::vector<EndgameTimings> buckets = play_endgames(dict, captured, params, thresholds);
    for (size_t k = 0; k < buckets.size(); ++k) {
      const EndgameTimings& t = buckets[k];
      if (t.games == 0) continue;
      const std::string label =
        k + 1 == buckets.size() ? "all" : bucket_label(static_cast<int>(k), thresholds);
      const double n = t.games;
      // The solver's endgame phase replaces the baseline's: E/B is the
      // endgame-phase multiplier, (A - B + E) / A the whole-game (self-play
      // throughput) multiplier.
      std::printf("%10llu %8s %7d %12.2f %15.2f %14.2f %10.2fx %8.2fx\n",
                  static_cast<unsigned long long>(b), label.c_str(), t.games, 1e3 * t.game_s / n,
                  1e3 * t.endgame_s / n, 1e3 * t.solver_s / n, t.solver_s / t.endgame_s,
                  (t.game_s - t.endgame_s + t.solver_s) / t.game_s);
    }
  }
}

using AgentFactory = std::function<std::unique_ptr<Agent>(int thread_id)>;

// Play `games` seeded games (seed base_seed+i) of factory-built agents and
// return the wall-clock seconds. Threads split the game indices into contiguous
// chunks; each thread builds its own pair of agents once and reuses them.
// `fast_track` makes the games respect agent projections, as self-play does.
double run_config(const Dictionary& dict, uint64_t base_seed, int games, int threads,
                  const AgentFactory& make0, const AgentFactory& make1, bool fast_track) {
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
        g.set_respect_projections(fast_track);
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

AgentFactory endgame_factory(const EndgameSolver::Params& params) {
  return [params](int tid) -> std::unique_ptr<Agent> {
    EndgameHastyBotAgent::Params p;
    p.hasty = HastyBotAgent::Params{.thread_id = tid, .name = "EndgameHastyBot"};
    p.solver = params;
    return std::make_unique<EndgameHastyBotAgent>(p);
  };
}

// Head-to-head aggregate for one bag-empty-spread bucket: the endgame bot's
// summed final spread plus its win/draw/loss record. Without spread_matters
// the spread shrinks by construction (winning positions stop maximizing it);
// the W/D/L record is the metric that must hold up.
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
// gives every configuration the same seat-independent conditioning variable
// for the head-to-head buckets.
int baseline_bag_empty_spread(const Dictionary& dict, uint64_t seed) {
  CapturedEndgame cap;
  bool captured = false;
  Clock::time_point endgame_start;
  FirstEndgameCapturer a0(0, "A", cap, captured, endgame_start);
  FirstEndgameCapturer a1(0, "B", cap, captured, endgame_start);
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
// not usable. Single-threaded so the per-game results are deterministic in
// `base_seed`.
std::vector<H2H> endgame_vs_hasty(const Dictionary& dict, uint64_t base_seed, int games,
                                  const EndgameSolver::Params& params,
                                  const std::vector<int>& thresholds) {
  const AgentFactory eg = endgame_factory(params);
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
                    const std::vector<uint64_t>& budgets, EndgameSolver::Params params,
                    const std::vector<int>& thresholds) {
  std::printf("games mode: %d games/config, threads=%d, plies=%d, spread-matters=%d\n\n", games,
              threads, params.plies, params.spread_matters ? 1 : 0);
  std::printf("%-22s %11s %10s %10s %10s %10s\n", "config", "budget", "total s", "s/game",
              "games/s", "ratio");

  const double base_s = run_config(dict, base_seed, games, threads, hasty_factory(),
                                   hasty_factory(), /*fast_track=*/false);
  print_games_row("hasty-vs-hasty", "-", base_s, games, 1.0);

  for (uint64_t b : budgets) {
    params.budget = b;
    const AgentFactory f = endgame_factory(params);
    const double s = run_config(dict, base_seed, games, threads, f, f, /*fast_track=*/true);
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
    params.budget = b;
    const std::vector<H2H> buckets = endgame_vs_hasty(dict, base_seed, games, params, thresholds);
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
    std::string mode = "endgames";
    int games = 100;
    uint64_t seed = 1;
    int threads = 1;
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
                       "games-mode parallelism (per-thread agents)");
    desc.add_options()(
      "spread-buckets", po::value<std::string>(&buckets_csv)->default_value(buckets_csv),
      "ascending |spread| thresholds bucketing endgames (and head-to-head games) by "
      "their bag-empty spread");
    desc.add_options()("leaves-file", po::value<std::string>(&leaves_file),
                       "path to leaves.klv2 (optional; defaults to the active lexicon's)");
    desc.add_options()("peg-file", po::value<std::string>(&peg_file)->default_value(""),
                       "path to preendgame.json (optional)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    scribblez::init_equity(leaves_file, peg_file);
    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    const std::vector<uint64_t> budgets = budgets_csv.empty()
                                            ? std::vector<uint64_t>{params.budget}
                                            : scribblez::parse_budgets(budgets_csv);
    const std::vector<int> thresholds = scribblez::parse_thresholds(buckets_csv);
    if (threads < 1) threads = 1;

    if (mode == "endgames") {
      scribblez::run_endgames_mode(dict, seed, games, budgets, params, thresholds);
    } else if (mode == "games") {
      scribblez::run_games_mode(dict, seed, games, threads, budgets, params, thresholds);
    } else {
      std::cerr << "unknown --mode '" << mode << "' (expected endgames or games)\n";
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
