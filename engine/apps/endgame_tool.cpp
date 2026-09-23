// endgame_tool: solves the endgame at the end of a GCG file and prints the
// solver's trace, for inspecting why the solver plays what it plays.
//
//   endgame_tool --gcg positions/NWL23/interesting-positions/foo.gcg
//   endgame_tool --gcg game.gcg --budget 50000000 --plies 12 --spread-matters false
//
// The GCG's final position must be a true endgame: the bag is empty and both
// racks are determinable (see read_gcg_endgame). The output is the position,
// the solver trace, the verdict, and the projected line.

#include "data/gcg_reader.h"
#include "data/gcg_writer.h"
#include "endgame/endgame_solver.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace scribblez {
namespace {

void run(const std::string& gcg_path, const EndgameSolver::Params& params) {
  std::ifstream in(gcg_path);
  if (!in.good()) throw util::CleanException("cannot read {}", gcg_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string gcg_text = buffer.str();

  ParsedGcgEndgame endgame;
  std::string error;
  if (!read_gcg_endgame(gcg_text, &endgame, &error))
    throw util::CleanException("GCG endgame lift failed: {}", error);
  const int mover = endgame.mover;

  const Dictionary& dict = load_dictionary_or_throw();

  std::cout << "position after " << endgame.turns << " turns (" << endgame.player_names[mover]
            << " to move):\n"
            << endgame.board.to_string() << "\n"
            << endgame.player_names[mover] << ": " << endgame.racks[mover].to_string() << ", "
            << endgame.scores[mover] << " points\n"
            << endgame.player_names[1 - mover] << ": " << endgame.racks[1 - mover].to_string()
            << ", " << endgame.scores[1 - mover] << " points\n\n";

  EndgameSolver solver;
  solver.set_trace(&std::cout, move_notation);
  const EndgameResult r =
    solver.solve({&dict, endgame.board, endgame.racks[mover], endgame.racks[1 - mover],
                  endgame.scores[mover], endgame.scores[1 - mover], /*scoreless_turns=*/0},
                 params);

  std::cout << "\nverdict: ";
  if (r.proven_class == EndgameResult::kClassUnknown) {
    std::cout << "class unproven";
  } else {
    std::cout << "proven "
              << (r.proven_class > 0   ? "WIN"
                  : r.proven_class < 0 ? "LOSS"
                                       : "DRAW")
              << " for " << endgame.player_names[mover];
  }
  std::cout << "; value " << r.value << (r.proven ? " (proven)" : " (estimate)") << ", depth "
            << r.depth_completed << ", nodes " << r.nodes << "\n";

  // The projected line, rendered against the evolving board.
  Board board = endgame.board;
  std::cout << "projection: " << move_notation(board, r.best) << "\n";
  board.apply(r.best);
  for (const Move& m : r.continuation) {
    std::cout << "            " << move_notation(board, m) << "\n";
    board.apply(m);
  }
  if (r.continuation.empty()) std::cout << "            (no certificate)\n";
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string gcg_path;
    // Analysis defaults (a generous budget, spread maximized) rather than the
    // agents' throughput-tuned ones.
    scribblez::EndgameSolver::Params params;
    params.budget = 1'000'000;
    params.spread_matters = true;

    po::options_description desc("endgame_tool options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("gcg", po::value<std::string>(&gcg_path)->required(),
                       "GCG file holding the endgame position (bag must be empty)");
    params.add_options(desc);
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);
    scribblez::run(gcg_path, params);
    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
