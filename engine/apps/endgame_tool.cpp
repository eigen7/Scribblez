// endgame_tool: solve one endgame position from a GCG file, with a verbose
// trace of the solver's reasoning.
//
// The GCG's final position must be a true endgame: the bag empty and both
// racks determinable. The mover's rack comes from a #Rack1/#Rack2 pragma (as
// the manual GCG tool writes) or, failing that, from the reader's tracked
// rack when it is fully known; the opponent's rack is derived as the tiles
// absent from both the board and the mover's rack. The trace shows the
// position, the solver's root block-or-outscore view (the replier's out-plays
// and every root move's futility bound), each deepening iteration, the
// certificate walk, and the final verdict with its projected line.
//
// Usage:
//   endgame_tool --gcg PATH [--budget N] [--plies P]
//                [--objective lexicographic|first-win|spread] [--lexicon NAME]

#include "data/gcg_reader.h"
#include "data/gcg_writer.h"
#include "endgame/endgame_solver.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/lexicon.h"
#include "selfplay/game_runner.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

// The rack recorded by a "#RackN TILES" pragma line ('?' is a blank), or
// nullopt when the file has none for that player. Matching is
// case-insensitive in the pragma name, as GCG writers vary.
std::optional<Rack> pragma_rack(const std::string& gcg_text, int player) {
  const std::string want = "#rack" + std::to_string(player + 1);
  std::istringstream lines(gcg_text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.size() < want.size() + 1) continue;
    std::string head = line.substr(0, want.size());
    for (char& c : head) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (head != want || line[want.size()] != ' ') continue;
    Rack rack;
    for (size_t i = want.size() + 1; i < line.size(); ++i) {
      const char c = line[i];
      if (c == ' ' || c == '\r') continue;
      rack.add(c == '?' ? BLANK : Tile::from_char(c));
    }
    return rack;
  }
  return std::nullopt;
}

// The tiles absent from both the board and `known_rack`: with an empty bag,
// exactly the opponent's rack. Throws when more than a rackful remain (the
// bag is not empty, so this is not an endgame position).
Rack derive_opponent_rack(const Board& board, const Rack& known_rack) {
  TileCounts remaining;
  for (int l = 0; l < 26; ++l)
    for (int i = 0; i < TILE_COUNTS[l]; ++i) remaining.add(Tile::of(l));
  for (int i = 0; i < TILE_COUNTS[BLANK]; ++i) remaining.add(BLANK);

  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (!g.has_letter()) continue;
      if (!remaining.remove(g.rack_tile()))
        throw Exception("board holds more copies of a tile than the distribution allows");
    }
  }
  for (int i = 0; i < known_rack.size(); ++i) {
    if (!remaining.remove(known_rack.tiles()[i]))
      throw Exception("the mover's rack holds a tile the distribution has run out of");
  }

  Rack opp;
  int count = 0;
  for (int l = 0; l <= 26; ++l) {
    const Tile t = l == 26 ? BLANK : Tile::of(l);
    for (int i = 0; i < remaining.count(t); ++i) {
      if (++count > RACK_SIZE)
        throw Exception(
          "more than a rackful of tiles is unaccounted for: the bag is not "
          "empty, so this is not an endgame position");
      opp.add(t);
    }
  }
  return opp;
}

std::string rack_string(const Rack& rack) {
  std::string s;
  for (int i = 0; i < rack.size(); ++i) s.push_back(rack.tiles()[i].to_char());
  return s;
}

void run(const std::string& gcg_path, uint64_t budget, int plies, EndgameObjective objective) {
  std::ifstream in(gcg_path);
  if (!in.good()) throw Exception("cannot read " + gcg_path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string gcg_text = buffer.str();

  ParsedGcgGame game;
  std::string error;
  if (!read_gcg_text(gcg_text, &game, &error)) throw Exception("GCG parse failed: " + error);
  if (game.snapshots.empty()) throw Exception("GCG contains no positions");

  const ParsedGcgSnapshot& snapshot = game.snapshots.back();
  const int mover = snapshot.turn_player;

  // A rack-slot array with unknown slots cannot distinguish an empty slot from
  // a hidden tile, so the mover's rack must come from an explicit pragma.
  const std::optional<Rack> mover_rack = pragma_rack(gcg_text, mover);
  if (!mover_rack.has_value())
    throw Exception("the mover's rack is unknown: add a #Rack" + std::to_string(mover + 1) +
                    " pragma to the GCG");

  std::optional<Rack> opp_rack = pragma_rack(gcg_text, 1 - mover);
  if (!opp_rack.has_value()) opp_rack = derive_opponent_rack(snapshot.board, *mover_rack);

  const Dictionary& dict = GameRunner::load_dictionary_or_throw();

  std::cout << "position after " << game.turns.size() << " turns (" << game.player_names[mover]
            << " to move):\n"
            << snapshot.board.to_string() << "\n"
            << game.player_names[mover] << ": " << rack_string(*mover_rack) << ", "
            << snapshot.scores[mover] << " points\n"
            << game.player_names[1 - mover] << ": " << rack_string(*opp_rack) << ", "
            << snapshot.scores[1 - mover] << " points\n\n";

  EndgameSolver solver;
  solver.set_trace(&std::cout, move_notation);
  const EndgameResult r =
    solver.solve(snapshot.board, dict, *mover_rack, *opp_rack, snapshot.scores[mover],
                 snapshot.scores[1 - mover], /*scoreless_turns=*/0, budget, plies, objective);

  std::cout << "\nverdict: ";
  if (r.proven_class == EndgameResult::kClassUnknown) {
    std::cout << "class unproven";
  } else {
    std::cout << "proven "
              << (r.proven_class > 0   ? "WIN"
                  : r.proven_class < 0 ? "LOSS"
                                       : "DRAW")
              << " for " << game.player_names[mover];
  }
  std::cout << "; value " << r.value << (r.proven ? " (proven)" : " (estimate)") << ", depth "
            << r.depth_completed << ", nodes " << r.nodes << "\n";

  // The projected line, rendered against the evolving board.
  Board board = snapshot.board;
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
    uint64_t budget = 1'000'000;
    int plies = 25;
    std::string objective_str = "lexicographic";

    po::options_description desc("endgame_tool options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("gcg", po::value<std::string>(&gcg_path)->required(),
                       "GCG file holding the endgame position (bag must be empty)");
    desc.add_options()("budget", po::value<uint64_t>(&budget)->default_value(budget),
                       "solver node budget");
    desc.add_options()("plies", po::value<int>(&plies)->default_value(plies),
                       "solver iterative-deepening depth cap");
    desc.add_options()("objective",
                       po::value<std::string>(&objective_str)->default_value(objective_str),
                       "solver objective: lexicographic | first-win | spread");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);
    scribblez::run(gcg_path, budget, plies, scribblez::parse_endgame_objective(objective_str));
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
