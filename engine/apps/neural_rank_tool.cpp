// neural_rank_tool: ranks the legal moves at one GCG position by the position
// evaluation model, exactly as NeuralAgent would score them, optionally beside
// a Monte-Carlo sim of the same moves. It is the tool for asking "what does
// the model think of this position, and is it right?"
//
//   neural_rank_tool --gcg pos.gcg --model teacher.onnx --top-k 20
//   neural_rank_tool --gcg game.gcg --turn 14 --model teacher.onnx -k 15 --sim
//
// Without --turn, the position is the file's final state, with the mover's rack
// taken from its #RackN pragma. With --turn N, it is the position before
// recorded turn N of an annotated game, with the rack that turn line records.
//
// Each candidate's post-move position is scored through the same
// CandidateEvaluator the agent uses. Candidates are every legal placement and
// exchange, or with --top-k K the K best by HastyBot static equity (the agent's
// own candidate cut). The table shows the model's win rate
// (P(win) + 0.5 P(draw)), its predicted final spread, and HastyBot equity.
// Exchanges rank alongside placements, since a post-exchange position is an
// ordinary input to the model; passes are never ranked. Whether the mover sees
// the opponent's retained leave follows the model's declared input arm, as it
// does for the agent.
//
// --sim sims every scored move with SimRunner (HastyBot rollouts under common
// random numbers) and adds its sim win rate and rank: the ground truth the
// model is judged against. Sim cost is rollouts x scored moves, so bound it with
// --top-k. Sims need tiles in the bag.

#include "agent/agent.h"
#include "agent/candidate_evaluator.h"
#include "agent/neural_service_options.h"
#include "data/gcg_reader.h"
#include "data/gcg_writer.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/eval_service.h"
#include "nn/model_specs.h"
#include "sim/sim_runner.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

struct Options {
  std::string gcg_path;
  int turn = 0;   // 1-based recorded turn; 0 = the file's final state
  int top_k = 0;  // moves scored: K>0 = top-K by static equity; 0 = all
  int rows = -1;  // rows printed; -1 = top_k (so 0 = every scored move)
  std::string objective = "winprob";
  bool exchanges = true;
  bool sim = false;
  SimRunner::Params sim_params{.rollouts = 300, .threads = util::default_thread_count()};
  uint64_t sim_seed = 1;
  NeuralServiceOptions service;
};

// One scored move: the model's verdict, its static equity, and (with --sim)
// its Monte-Carlo win rate.
struct RankedMove {
  int index;         // into the legal-move list
  float win_rate;    // the WLD head's P(win) + 0.5 P(draw)
  float score_diff;  // the ScoreDiff head's predicted mean final spread
  float objective;   // the value the ranking sorts on
  double equity;
  int equity_rank;  // 1 = HastyBot's greedy choice
  double sim_win_rate = 0;
  int sim_rank = 0;  // 1 = the sim's best among the scored moves
};

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) throw util::CleanException("cannot read {}", path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// `open_leaves` is true when the served model's input arm lets the mover see
// the opponent's retained leave.
ParsedGcgPosition read_position(const Options& opt, bool open_leaves) {
  const std::string gcg_text = read_file(opt.gcg_path);
  ParsedGcgPosition pos;
  std::string error;
  const bool ok = opt.turn == 0
                    ? read_gcg_position(gcg_text, open_leaves, &pos, &error)
                    : read_gcg_position_at(gcg_text, opt.turn - 1, open_leaves, &pos, &error);
  if (!ok) {
    throw util::CleanException("GCG position lift failed (--turn {} = turn index {}): {}", opt.turn,
                               opt.turn - 1, error);
  }
  return pos;
}

// An evaluator that has observed `pos`'s recorded moves in turn order, the same
// state the agent would be in during a live game.
CandidateEvaluator replayed_evaluator(const Dictionary& dict,
                                      std::shared_ptr<nn::PositionEvalService> service,
                                      int max_batch, const ParsedGcgPosition& pos) {
  CandidateEvaluator evaluator(dict, std::move(service), max_batch);
  evaluator.begin_game(BeginGameRequest{});
  for (const ParsedGcgTurn& t : pos.game.turns) evaluator.observe_move(t.record.move);
  RELEASE_ASSERT(evaluator.active_player() == pos.mover);
  return evaluator;
}

std::vector<Move> legal_moves(const MoveRequest& req, bool exchanges) {
  std::vector<Move> moves = generate_legal_plays(req);
  if (!exchanges) return moves;
  const std::vector<Move> swaps = generate_legal_exchanges(req);
  moves.insert(moves.end(), swaps.begin(), swaps.end());
  return moves;
}

// Move indices, best static equity first: the order whose head NeuralAgent's
// top-K cut keeps.
std::vector<int> equity_order(const std::vector<double>& equities) {
  std::vector<int> order(equities.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return equities[size_t(a)] > equities[size_t(b)]; });
  return order;
}

// Score the top_k (0 = all) moves by static equity the way the agent does,
// then sort them best-first by `objective`.
std::vector<RankedMove> rank_moves(CandidateEvaluator& evaluator, const MoveRequest& req,
                                   const std::vector<Move>& moves, int top_k,
                                   EvalObjective objective) {
  const std::vector<double> equities =
    HastyEquity::instance().equities(moves, req.board, req.bag_size, req.opp_rack, req.my_rack);
  std::vector<int> idx = equity_order(equities);
  if (top_k > 0 && top_k < int(idx.size())) idx.resize(size_t(top_k));
  const int k = idx.size();
  evaluator.evaluate(req, moves, idx, k);

  std::vector<RankedMove> ranked;
  ranked.reserve(size_t(k));
  for (int i = 0; i < k; ++i) {
    const float* wld = evaluator.wld_row(i);
    const float* sd = evaluator.score_diff_row(i);
    ranked.push_back({.index = idx[size_t(i)],
                      .win_rate = objective_value(wld, sd, EvalObjective::kWinProb),
                      .score_diff = sd[0],
                      .objective = objective_value(wld, sd, objective),
                      .equity = equities[size_t(idx[size_t(i)])],
                      .equity_rank = i + 1});
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const RankedMove& a, const RankedMove& b) {
    return a.objective > b.objective;
  });
  return ranked;
}

// 1-based rank of each ranked move by sim win rate, ties going to the better
// model rank.
void assign_sim_ranks(std::vector<RankedMove>& ranked) {
  std::vector<int> order(ranked.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return ranked[size_t(a)].sim_win_rate > ranked[size_t(b)].sim_win_rate;
  });
  for (size_t r = 0; r < order.size(); ++r) ranked[size_t(order[r])].sim_rank = int(r) + 1;
}

void sim_ranked_moves(const Dictionary& dict, const ParsedGcgPosition& pos,
                      const std::vector<Move>& moves, const Options& opt,
                      std::vector<RankedMove>& ranked) {
  if (pos.bag_size == 0) throw util::CleanException("--sim needs tiles in the bag");
  std::vector<Move> candidates;
  candidates.reserve(ranked.size());
  for (const RankedMove& r : ranked) candidates.push_back(moves[size_t(r.index)]);

  std::cout << "simming " << candidates.size() << " moves x " << opt.sim_params.rollouts
            << " rollouts on " << opt.sim_params.threads << " threads...\n";
  const SimRunner runner(dict, opt.sim_params);
  const std::vector<SimObservation> observations = runner.run(
    {pos.board, pos.scores, pos.mover, pos.rack, pos.opp_leave}, candidates, opt.sim_seed);
  for (size_t i = 0; i < ranked.size(); ++i) {
    ranked[i].sim_win_rate = sim_objective_value(observations[i], SimObjective::kWinRate);
  }
  assign_sim_ranks(ranked);
}

void print_position(const ParsedGcgPosition& pos) {
  const int mover = pos.mover;
  const auto& names = pos.game.player_names;
  std::cout << "position after " << pos.turns << " turns (" << names[mover] << " to move):\n"
            << pos.board.to_string() << "\n"
            << names[mover] << ": " << pos.rack.to_string() << ", " << pos.scores[mover]
            << " points\n"
            << names[1 - mover] << ": " << pos.scores[1 - mover] << " points";
  if (!pos.opp_leave.empty()) std::cout << ", known leave " << pos.opp_leave.to_string();
  std::cout << "\nbag: " << pos.bag_size << " tiles\n";
  if (pos.bag_size == 0) {
    std::cout << "note: the bag is empty; the model never trains on endgame positions and "
                 "NeuralAgent hands these to the exact solver instead\n";
  }
  std::cout << "\n";
}

// The first `rows` (0 = all) of `ranked`; the sim columns only when sims ran.
void print_table(const Board& board, const std::vector<Move>& moves,
                 const std::vector<RankedMove>& ranked, int rows, bool sim) {
  const int shown = rows == 0 ? int(ranked.size()) : std::min<int>(rows, ranked.size());
  std::printf("%5s  %-28s %5s  %6s  %7s  %12s %10s", "rank", "move", "score", "win%", "spread",
              "hasty_equity", "hasty_rank");
  if (sim) std::printf("  %8s %8s", "sim_win%", "sim_rank");
  std::printf("\n");
  for (int r = 0; r < shown; ++r) {
    const RankedMove& p = ranked[size_t(r)];
    const Move& mv = moves[size_t(p.index)];
    std::printf("%5d  %-28s %5d  %6.2f  %+7.1f  %12.1f %10d", r + 1,
                move_notation(board, mv).c_str(), int(mv.score()), 100.0f * p.win_rate,
                p.score_diff, p.equity, p.equity_rank);
    if (sim) std::printf("  %8.2f %8d", 100.0 * p.sim_win_rate, p.sim_rank);
    std::printf("\n");
  }
}

void validate(const Options& opt) {
  if (opt.turn < 0) throw util::CleanException("--turn must be >= 1 (0 = the final position)");
  if (opt.top_k < 0) throw util::CleanException("--top-k must be >= 0 (0 = all moves)");
  if (opt.rows < -1) throw util::CleanException("--rows must be >= 0 (0 = every scored move)");
  if (opt.sim) SimRunner::validate(opt.sim_params);
}

void run(const Options& opt) {
  validate(opt);
  const int rows = opt.rows == -1 ? opt.top_k : opt.rows;
  const EvalObjective objective = parse_eval_objective(opt.objective, "--objective");
  const nn::NeuralNetParams<nn::PositionEvaluationSpec> net_params =
    opt.service.net_params<nn::PositionEvaluationSpec>(0);

  std::shared_ptr<nn::PositionEvalService> service = nn::PositionEvalService::create(net_params);
  const ParsedGcgPosition pos = read_position(opt, service->opp_leave_input());
  print_position(pos);

  const Dictionary& dict = load_dictionary_or_throw();
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const MoveRequest req{
    pos.board,   dict, pos.rack, pos.opp_leave, pos.scores[pos.mover], pos.scores[1 - pos.mover],
    pos.bag_size};
  const std::vector<Move> moves = legal_moves(req, opt.exchanges);
  if (moves.empty()) {
    std::cout << "no legal move but a pass\n";
    return;
  }

  CandidateEvaluator evaluator =
    replayed_evaluator(dict, std::move(service), net_params.max_rows, pos);
  std::vector<RankedMove> ranked = rank_moves(evaluator, req, moves, opt.top_k, objective);
  if (opt.sim) sim_ranked_moves(dict, pos, moves, opt, ranked);
  std::cout << moves.size() << " legal moves, " << ranked.size() << " scored, ranked by "
            << opt.objective << ":\n";
  print_table(pos.board, moves, ranked, rows, opt.sim);
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    scribblez::Options opt;
    po::options_description desc("neural_rank_tool options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("gcg", po::value<std::string>(&opt.gcg_path)->required(),
                       "GCG file holding the position");
    desc.add_options()(
      "turn", po::value<int>(&opt.turn)->default_value(opt.turn),
      "rank the position before this recorded turn (1-based), holding the rack that turn line "
      "records; 0 = the file's final state, whose mover's rack comes from its #RackN pragma");
    desc.add_options()(
      "top-k,k", po::value<int>(&opt.top_k)->default_value(opt.top_k),
      "moves scored by the model: K>0 = the top-K by HastyBot static equity (NeuralAgent's "
      "candidate cut); 0 = every legal move");
    desc.add_options()("rows,r", po::value<int>(&opt.rows),
                       "rows printed (default: --top-k, so 0 = every scored move)");
    desc.add_options()(
      "objective,o", po::value<std::string>(&opt.objective)->default_value(opt.objective),
      "ranking head: winprob = P(win)+0.5*P(draw); scorediff = expected final spread");
    desc.add_options()(
      "exchanges", po::value<bool>(&opt.exchanges)->default_value(opt.exchanges),
      "rank legal exchanges together with placements (0|1); 0 ranks placements only");
    desc.add_options()("sim", po::bool_switch(&opt.sim),
                       "also Monte-Carlo sim every scored move (HastyBot rollouts to game end) "
                       "and print its sim win rate and rank");
    desc.add_options()(
      "sim-rollouts",
      po::value<int>(&opt.sim_params.rollouts)->default_value(opt.sim_params.rollouts),
      "--sim: rollouts per scored move");
    desc.add_options()(
      "sim-threads", po::value<int>(&opt.sim_params.threads)->default_value(opt.sim_params.threads),
      "--sim: rollout worker threads");
    desc.add_options()("sim-seed", po::value<uint64_t>(&opt.sim_seed)->default_value(opt.sim_seed),
                       "--sim: base seed of the common random numbers");
    opt.service.add_options(desc);
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);
    scribblez::run(opt);
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
