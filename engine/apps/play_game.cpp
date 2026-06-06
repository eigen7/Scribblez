// play_game: runs Scrabble games between two agents (Greedy, HastyBot, or
// Human) and optionally writes one GCG-format game log per game to a
// directory. A Human player is driven through a local web UI.
//
// Usage:
//   play_game [--player "--type=T [opts]"]... [--lexicon NAME] [--seed N]
//             [--log-dir DIR] [--games N] [--threads N] [--verbose]
//
// Each --player spec selects a seat; repeat once per seat (defaults to two
// greedy players). The human agent's own --port / --vite-port / --web-dir
// options live inside its --player spec, e.g.
//   --player "--type=human --port=8081 --web-dir=web"
//
// For human play the engine launches the front-end's Vite dev server itself
// (npm run dev) and opens the browser at it -- you never run npm by hand. Run
// ./build.py once first to install the web dependencies.

#include "scribblez/cli.h"
#include "scribblez/exception.h"
#include "scribblez/game_runner.h"
#include "scribblez/lexicon.h"
#include "scribblez/macondo_oracle_pool.h"
#include "scribblez/player_factory.h"
#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    // Each subsystem owns its own Params + add_options() so this top-level
    // function never has to know which knobs belong to whom.
    scribblez::SeedProducer::Params seed_params;
    scribblez::PlayerFactory::Params player_params;
    scribblez::GameRunner::Params runner_params;

    po::options_description desc("play_game options");
    desc.add_options()("help,h", "show this help message and exit");
    scribblez::Lexicon::instance().add_options(desc);
    scribblez::MacondoOraclePool::instance().add_options(desc);
    seed_params.add_options(desc);
    player_params.add_options(desc);
    runner_params.add_options(desc);

    scribblez::parse_command_line(
        argc, argv, desc,
        "Player types (use --player \"--type=X [options]\"):\n\n" +
            scribblez::PlayerFactory::all_player_types_help());

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
