// play_game: plays games between two agents, optionally logging each game as
// GCG (--log-dir) or into .slog training files (--binary-log-dir). This is the
// general match and self-play driver; `--help` lists every player type and
// runner option.
//
//   play_game --player "--type=hastybot" --player "--type=greedy" --games 100 --threads 8
//
// Give two --player specs, or none for two greedy seats. Each agent's own
// options go inside its spec. A human seat is played in the browser: the engine
// starts the web UI's Vite dev server itself and opens it, so run py/build.py
// once beforehand to install the web dependencies.
//
//   play_game --player "--type=human --port=8081" --player "--type=hastybot"

#include "agent/player_factory.h"
#include "arena/game_runner.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/misc.h"
#include "util/seed_producer.h"

#include <boost/program_options.hpp>

#include <string>

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
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
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
