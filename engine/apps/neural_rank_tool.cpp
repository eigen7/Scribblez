// neural_rank_tool: rank every legal placement at one GCG position by the
// position evaluation model, the way NeuralAgent ranks its candidates.
//
// The position is either the file's final recorded state, with the mover's
// rack taken from its #RackN pragma (read_gcg_position), or -- for a complete
// annotated game -- the position before recorded turn N, with the rack that
// turn line records (--turn N, read_gcg_position_at). Every legal placement is
// applied, its post-move position encoded from the mover's POV and scored by
// the model through the same CandidateEvaluator the agent drives, and the
// top K by the chosen objective are printed with the model's win/draw/loss
// probabilities, its predicted final spread, and HastyBot static equity for
// comparison. Exchanges and passes are not ranked: the model trains on
// post-placement positions only, which is also why NeuralAgent scores none.
//
// Usage:
//   neural_rank_tool --gcg PATH --model PATH.onnx [--turn N] [--top-k K]
//                    [--objective winprob|scorediff] [--open-leaves]
//                    [--batch-size B] [--cuda-device D] [--precision P]
//                    [--lexicon NAME]

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
#include "util/assert.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <algorithm>
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
  int turn = 0;  // 1-based recorded turn; 0 = the file's final state
  int top_k = 20;
  std::string objective = "winprob";
  bool open_leaves = false;
  NeuralServiceOptions service;
};

// One legal placement's model verdict, alongside its static equity.
struct RankedPlay {
  int index;  // into the legal-play list
  float win, draw, loss;
  float score_diff;  // the ScoreDiff head's predicted mean final spread
  float objective;   // the value the ranking sorts on
  double equity;
  int equity_rank = 0;  // 1 = HastyBot's greedy choice
};

std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) throw util::CleanException("cannot read {}", path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

ParsedGcgPosition read_position(const Options& opt) {
  const std::string gcg_text = read_file(opt.gcg_path);
  ParsedGcgPosition pos;
  std::string error;
  const bool ok = opt.turn == 0
                    ? read_gcg_position(gcg_text, opt.open_leaves, &pos, &error)
                    : read_gcg_position_at(gcg_text, opt.turn - 1, opt.open_leaves, &pos, &error);
  if (!ok) {
    throw util::CleanException("GCG position lift failed (--turn {} = turn index {}): {}", opt.turn,
                               opt.turn - 1, error);
  }
  return pos;
}

// An evaluator whose mirrored game is `pos`: the recorded moves replayed in
// turn order, as the agent observes them during a live game.
CandidateEvaluator replayed_evaluator(const Dictionary& dict,
                                      std::shared_ptr<nn::PositionEvalService> service,
                                      int max_batch, const ParsedGcgPosition& pos) {
  CandidateEvaluator evaluator(dict, std::move(service), max_batch);
  evaluator.begin_game(BeginGameRequest{});
  for (const ParsedGcgTurn& t : pos.game.turns) evaluator.observe_move(t.record.move);
  RELEASE_ASSERT(evaluator.active_player() == pos.mover);
  return evaluator;
}

void assign_equity_ranks(std::vector<RankedPlay>& plays) {
  std::vector<int> order(plays.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return plays[size_t(a)].equity > plays[size_t(b)].equity; });
  for (size_t r = 0; r < order.size(); ++r) plays[size_t(order[r])].equity_rank = int(r) + 1;
}

// Every play scored by the model (in one chunked batch, as the agent scores
// its candidates) and by static equity, sorted best-first by `objective`.
std::vector<RankedPlay> rank_plays(CandidateEvaluator& evaluator, const MoveRequest& req,
                                   const std::vector<Move>& plays, EvalObjective objective) {
  const int n = plays.size();
  std::vector<int> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  evaluator.evaluate(req, plays, idx, n);
  const std::vector<double> equities =
    HastyEquity::instance().equities(plays, req.board, req.bag_size, req.opp_rack, req.my_rack);

  std::vector<RankedPlay> ranked;
  ranked.reserve(size_t(n));
  for (int i = 0; i < n; ++i) {
    const float* wld = evaluator.wld_row(i);
    const float* sd = evaluator.score_diff_row(i);
    ranked.push_back({.index = i,
                      .win = wld[0],
                      .draw = wld[1],
                      .loss = wld[2],
                      .score_diff = sd[0],
                      .objective = objective_value(wld, sd, objective),
                      .equity = equities[size_t(i)]});
  }
  assign_equity_ranks(ranked);
  std::stable_sort(ranked.begin(), ranked.end(), [](const RankedPlay& a, const RankedPlay& b) {
    return a.objective > b.objective;
  });
  return ranked;
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

void print_table(const Board& board, const std::vector<Move>& plays,
                 const std::vector<RankedPlay>& ranked, int top_k) {
  const int shown = std::min<int>(top_k, ranked.size());
  std::printf("%4s  %-28s %5s  %6s %6s %6s  %7s  %7s %4s\n", "rank", "play", "score", "win%",
              "draw%", "loss%", "spread", "equity", "eq#");
  for (int r = 0; r < shown; ++r) {
    const RankedPlay& p = ranked[size_t(r)];
    const Move& mv = plays[size_t(p.index)];
    std::printf("%4d  %-28s %5d  %6.2f %6.2f %6.2f  %+7.1f  %7.1f %4d\n", r + 1,
                move_notation(board, mv).c_str(), int(mv.score()), 100.0f * p.win, 100.0f * p.draw,
                100.0f * p.loss, p.score_diff, p.equity, p.equity_rank);
  }
}

void run(const Options& opt) {
  if (opt.turn < 0) throw util::CleanException("--turn must be >= 1 (0 = the final position)");
  if (opt.top_k < 1) throw util::CleanException("--top-k must be >= 1");
  const EvalObjective objective = parse_eval_objective(opt.objective, "--objective");
  const nn::NeuralNetParams<nn::PositionEvaluationSpec> net_params =
    opt.service.net_params<nn::PositionEvaluationSpec>(0);

  const ParsedGcgPosition pos = read_position(opt);
  print_position(pos);

  const Dictionary& dict = load_dictionary_or_throw();
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const MoveRequest req{
    pos.board,   dict, pos.rack, pos.opp_leave, pos.scores[pos.mover], pos.scores[1 - pos.mover],
    pos.bag_size};
  const std::vector<Move> plays = generate_legal_plays(req);
  if (plays.empty()) {
    std::cout << "no legal placement\n";
    return;
  }

  CandidateEvaluator evaluator =
    replayed_evaluator(dict, nn::PositionEvalService::create(net_params), net_params.max_rows, pos);
  const std::vector<RankedPlay> ranked = rank_plays(evaluator, req, plays, objective);
  std::cout << plays.size() << " legal placements, ranked by " << opt.objective << ":\n";
  print_table(pos.board, plays, ranked, opt.top_k);
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
    desc.add_options()("top-k,k", po::value<int>(&opt.top_k)->default_value(opt.top_k),
                       "placements to print");
    desc.add_options()(
      "objective,o", po::value<std::string>(&opt.objective)->default_value(opt.objective),
      "ranking head: winprob = P(win)+0.5*P(draw); scorediff = expected final spread");
    desc.add_options()("open-leaves", po::bool_switch(&opt.open_leaves),
                       "face-up leaves: the opponent's retained leave is known to the mover");
    opt.service.add_options(desc);
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);
    scribblez::run(opt);
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
