// pre_endgame_rank_tool: the pre-endgame solver's standing of every play at
// positions of recorded games with a few tiles in the bag
// (endgame/pre_endgame_solver.h): how many of the bag's possible contents
// each play wins against an opponent who replies with full knowledge. It is
// the referee for what an agent plays there
// (py/scripts/deep_bot/blind_spot_picks.py --peg-ranks): a sim whose opponent
// plays HastyBot's move overrates a play that only works unanswered.
//
//   pre_endgame_rank_tool --positions list.txt --out ranks.jsonl
//
// --positions names a file of lines "<.slog path> <game> <turn>", both
// 0-based (position_pick_tool's). --out gets a JSON line per position, in the
// file's order: its "slog", "game" and "turn", the "bag", the "total" bag
// draws each play is scored over, and "plays", best first, each a "move" in
// the reports' notation (data/gcg_writer.h's move_notation) with the "points"
// it wins of them, a draw counting half. The mover's rack is the replay's; the
// opponent's is hidden, as in play.

#include "arena/recorded_turn.h"
#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "endgame/endgame_oracle.h"
#include "endgame/endgame_solver.h"
#include "endgame/pre_endgame_solver.h"
#include "game/game_log.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/work_pool.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace scribblez {
namespace {

namespace json = boost::json;

struct Options {
  std::string positions;
  std::string out;
  uint64_t budget = 2000;
  PreEndgameSolver::Params solver{};
  int threads = util::default_thread_count();
};

void add_options(boost::program_options::options_description& desc, Options& opt) {
  namespace po = boost::program_options;
  desc.add_options()("help,h", "show this help message and exit")(
    "positions", po::value<std::string>(&opt.positions),
    "a file of lines \"<.slog path> <game> <turn>\" (0-based)")(
    "out", po::value<std::string>(&opt.out), "the JSON-lines file to write (default: stdout)")(
    "budget", po::value<uint64_t>(&opt.budget)->default_value(opt.budget),
    "node budget for each endgame the solver evaluates")(
    "threads", po::value<int>(&opt.threads)->default_value(opt.threads),
    "CPU threads; positions are solved in parallel, one thread each");
  opt.solver.add_options(desc, "");
}

void validate(const Options& opt) {
  if (opt.positions.empty()) throw util::CleanException("--positions is required");
  if (opt.threads < 1) throw util::CleanException("--threads must be >= 1");
}

EndgameSolver& thread_solver() {
  thread_local EndgameSolver solver;
  return solver;
}

json::array plays_json(const Board& board, const std::vector<PreEndgameSolver::RankedPlay>& plays) {
  json::array out;
  for (const PreEndgameSolver::RankedPlay& p : plays)
    out.push_back(json::object{{"move", move_notation(board, p.move)}, {"points", p.points}});
  return out;
}

// The solver's ranking at `a`, a position of the loaded .slog `buf`, as its
// line.
std::string rank(const RecordedTurnRef& a, const std::vector<char>& buf, const Dictionary& dict,
                 const Options& opt) {
  std::vector<TurnRecord> scratch;
  const GameLog g = binlog::make_game_view(buf.data(), a.game, scratch, nullptr);
  const RecordedPosition at = recorded_position(g, a.turn);
  if (at.bag < 1 || at.bag > PreEndgameSolver::kMaxInBag)
    throw util::CleanException(
      "{} game {} turn {}: {} tiles in the bag, and the solver takes 1 to "
      "{}",
      a.slog, a.game, a.turn, at.bag, PreEndgameSolver::kMaxInBag);
  thread_solver().clear();
  SolverEndgameOracle oracle(thread_solver(), opt.budget);
  const PreEndgamePosition pos{&dict,
                               at.board,
                               at.racks[at.mover],
                               Rack{},
                               at.scores[at.mover],
                               at.scores[1 - at.mover],
                               at.scoreless_turns};
  const std::vector<PreEndgameSolver::RankedPlay> plays =
    PreEndgameSolver(oracle).solve(pos, opt.solver);
  return json::serialize(json::object{{"slog", a.slog},
                                      {"game", a.game},
                                      {"turn", a.turn},
                                      {"bag", at.bag},
                                      {"total", plays.empty() ? 0 : plays.front().total},
                                      {"plays", plays_json(at.board, plays)}});
}

void run(const Options& opt) {
  const std::vector<RecordedTurnRef> asked = read_recorded_turn_refs(opt.positions);
  const std::map<std::string, std::vector<char>> slogs = load_recorded_slogs(asked);
  const Dictionary& dict = load_dictionary_or_throw();
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  std::vector<std::string> lines(asked.size());
  util::parallel_for(
    asked.size(), [&](size_t i) { lines[i] = rank(asked[i], slogs.at(asked[i].slog), dict, opt); });
  std::ofstream file;
  if (!opt.out.empty()) {
    file.open(opt.out);
    if (!file) throw util::CleanException("cannot write {}", opt.out);
  }
  std::ostream& out = opt.out.empty() ? std::cout : file;
  for (const std::string& line : lines) out << line << "\n";
  std::cerr << std::format("ranked {} position(s)\n", asked.size());
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  using namespace scribblez;
  try {
    Options opt;
    boost::program_options::options_description desc("pre_endgame_rank_tool options");
    add_options(desc, opt);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);
    util::set_worker_count(opt.threads);
    run(opt);
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
