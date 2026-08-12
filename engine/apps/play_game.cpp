// play_game: runs Scrabble games between two agents (Greedy, HastyBot, or
// Human) and optionally writes one GCG-format game log per game to a
// directory. A Human player is driven through a local web UI.
//
// Usage:
//   play_game [--player "--type=T [opts]"]... [--lexicon NAME] [--seed N]
//             [--log-dir DIR] [--games N] [--threads N] [--verbose]
//             [--leaves-file PATH] [--peg-file PATH]
//
// Each --player spec selects a seat; repeat once per seat (defaults to two
// greedy players). The human agent's own --port / --vite-port / --web-dir
// options live inside its --player spec, e.g.
//   --player "--type=human --port=8081 --web-dir=web"
//
// For human play the engine launches the front-end's Vite dev server itself
// (npm run dev) and opens the browser at it -- you never run npm by hand. Run
// py/build.py once first to install the web dependencies.

#include "agent/player_factory.h"
#include "arena/game_runner.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    // Each subsystem owns its own Params + add_options() so this top-level
    // function never has to know which knobs belong to whom.
    scribblez::SeedProducer::Params seed_params;
    scribblez::PlayerFactory::Params player_params;
    scribblez::GameRunner::Params runner_params;

    std::string leaves_file;
    std::string peg_file;

    po::options_description desc("play_game options");
    desc.add_options()("help,h", "show this help message and exit");
    desc.add_options()("leaves-file", po::value<std::string>(&leaves_file),
                       "path to leaves.klv2 (optional; HastyBot otherwise loads the "
                       "default leaves for the active lexicon)");
    desc.add_options()("peg-file", po::value<std::string>(&peg_file)->default_value(""),
                       "path to preendgame.json adjustment table (optional; omit to skip "
                       "pre-endgame equity adjustment)");
    scribblez::Lexicon::instance().add_options(desc);
    seed_params.add_options(desc);
    player_params.add_options(desc);
    runner_params.add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc,
                                        "Player types (use --player \"--type=X [options]\"):\n\n" +
                                          scribblez::PlayerFactory::all_player_types_help());

    if (!leaves_file.empty()) {
      scribblez::HastyEquity::init(leaves_file, peg_file);
    }

    scribblez::SeedProducer::instance().seed(seed_params);

    scribblez::GameRunner runner(runner_params, player_params);
    runner.run();
    return 0;
  } catch (const scribblez::CleanExit&) {
    return 0;
  } catch (const scribblez::Exception&) {
    return 1;
  } catch (const std::exception& e) {  // unexpected
    std::cerr << "Unexpected error: " << e.what() << "\n";
    return 1;
  }
}
