// play_game: runs a single Scrabble game between two agents (Greedy or Human)
// and writes a JSON log. A Human player is driven through a local web UI.
//
// Usage:
//   play_game [--player "--type=T"]... [--kwg <path>] [--seed N]
//             [--out game.json] [--port N] [--web-dir DIR] [--vite-port N]
//             [--verbose]
//   where each --player spec selects a seat, e.g.
//   --player "--type=human" --player "--type=greedy"
//   (repeat once per seat; defaults to two greedy players, at most one human).
//
// For human play the engine launches the front-end's Vite dev server itself
// (npm run dev) and opens the browser at it -- you never run npm by hand. Run
// ./build.py once first to install the web dependencies.

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/gcg_writer.h"
#include "scribblez/player_factory.h"
#include "scribblez/seed_producer.h"
#include "scribblez/web_server.h"

#include <boost/program_options.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// Default lexicon location, relative to the current working directory. The .kwg
// binary is not committed (it encodes a copyrighted wordlist); place or symlink
// it here, or point --kwg elsewhere.
constexpr const char* kDefaultKwg = "data/lexica/NWL23.kwg";
// Front-end package directory (containing package.json). The engine runs
// `npm run dev` here for human play.
constexpr const char* kDefaultWebDir = "web";

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;

  std::string kwg_path, dict_path, out_path, web_dir;
  std::vector<std::string> player_specs;
  int port = 0, vite_port = 0, games = 1;
  uint64_t seed = 0;
  bool verbose = false;

  // One option per statement so the declarations stay readable (chaining the
  // operator() calls together formats into an unreadable blob).
  po::options_description desc("play_game options");
  auto opt = desc.add_options();
  opt("help,h", "show this help message and exit");
  opt("player", po::value<std::vector<std::string>>(&player_specs)->composing(),
      "add a seat, e.g. --player \"--type=human\" --player \"--type=greedy\" "
      "(repeat once per seat; default: two greedy, at most one human)");
  opt("kwg", po::value<std::string>(&kwg_path)->default_value(kDefaultKwg),
      "lexicon .kwg file to load");
  opt("dict", po::value<std::string>(&dict_path), "alias for --kwg");
  opt("seed", po::value<uint64_t>(&seed), "PRNG seed (default: hardware random)");
  opt("out", po::value<std::string>(&out_path), "write the game-log JSON here (default: stdout)");
  opt("port", po::value<int>(&port)->default_value(8080), "engine WebSocket port");
  opt("vite-port", po::value<int>(&vite_port)->default_value(5173), "browser UI (Vite) port");
  opt("web-dir", po::value<std::string>(&web_dir)->default_value(kDefaultWebDir),
      "front-end package dir, used only when a human plays");
  opt("games", po::value<int>(&games)->default_value(1),
      "play this many games in one process (seeds seed, seed+1, ...); no human");
  opt("verbose,v", po::bool_switch(&verbose), "print final score and turn count to stderr");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
    return 2;
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    std::cout << "Player types (use --player \"--type=X [options]\"):\n\n"
              << scribblez::all_player_types_help();
    return 0;
  }
  // --dict is an accepted alias for --kwg.
  if (vm.count("dict")) kwg_path = dict_path;
  const bool seed_given = vm.count("seed") > 0;

  // Resolve the two player seats, defaulting to two greedy players. Each spec
  // is parsed by the player factory (e.g. --player "--type=human").
  if (player_specs.empty()) {
    player_specs = {"--type=greedy", "--type=greedy"};
  }
  if (player_specs.size() != 2) {
    std::cerr << "Expected exactly two --player specs (got " << player_specs.size() << ").\n";
    return 2;
  }
  std::array<scribblez::PlayerSpec, 2> players;
  int human_seat = -1;
  for (int s = 0; s < 2; ++s) {
    try {
      players[s] = scribblez::parse_player_spec(player_specs[s]);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 2;
    }
    if (players[s].is_human()) {
      if (human_seat >= 0) {
        std::cerr << "At most one human player is supported.\n";
        return 2;
      }
      human_seat = s;
    }
  }
  if (games < 1) {
    std::cerr << "--games must be >= 1\n";
    return 2;
  }
  if (games > 1 && human_seat >= 0) {
    std::cerr << "--games > 1 is for bot-only batches; a human plays one game.\n";
    return 2;
  }

  // Eager validation for non-human seats: construct (and discard) the agent
  // once so option errors surface here, before we resolve the seed or launch
  // the web UI. Done BEFORE SeedProducer is seeded so any seeds this consumes
  // don't shift the deterministic sequence the real agents will receive.
  // (Human seats need a live WebSession to construct, so we let those errors
  // surface lazily in the game loop.)
  for (int s = 0; s < 2; ++s) {
    if (players[s].is_human()) continue;
    try {
      auto a = scribblez::make_player(players[s], nullptr, players[1 - s].display_name());
      (void)a;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 2;
    }
  }

  // Resolve the seed (the lexicon path already defaulted via program_options).
  if (!seed_given) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }
  // Seed the process-wide SeedProducer so every RNG-using object constructed
  // after this point (e.g. GreedyAgent's tie-breaker) is deterministic when
  // --seed was given. Per-agent --seed options still override locally.
  scribblez::SeedProducer::instance().seed(seed);

  scribblez::Dictionary dict;
  try {
    dict = scribblez::Dictionary::load_kwg(kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << kwg_path
              << "'. Place or symlink an NWL "
                 ".kwg there, or pass --kwg <path>.\n";
    return 1;
  }
  if (verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from " << kwg_path << "\n";
    std::cerr << "Seed: " << seed << "\n";
  }

  // Stand up the web server up front (if a human is playing) so the browser can
  // connect before the game loop reaches the human's first turn. The engine
  // also launches the front-end's Vite dev server itself and opens the browser
  // at it -- so no npm commands need to be run by hand.
  std::unique_ptr<scribblez::WebSession> session;
  std::unique_ptr<scribblez::ViteDevServer> vite;
  if (human_seat >= 0) {
    try {
      session = std::make_unique<scribblez::WebSession>(port);
      vite = std::make_unique<scribblez::ViteDevServer>(web_dir, vite_port, port);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 1;
    }
    std::cerr << "\n  Starting the web UI (npm run dev in " << web_dir << ")...\n";
    if (!vite->wait_until_ready()) {
      std::cerr << "Error: the Vite dev server did not start. See " << web_dir
                << "/.vite-dev.log for details.\n"
                << "Did you run ./build.py to install the web dependencies?\n";
      return 1;
    }
    std::cerr << "\n  Human-vs-AI game ready.\n"
              << "  Open  " << vite->url() << "  in your browser to play.\n\n";
    std::string cmd = "xdg-open " + vite->url() + " >/dev/null 2>&1 &";
    int rc = std::system(cmd.c_str());  // best-effort; ignore failure
    (void)rc;
  }

  auto make_agent = [&](int seat) -> std::unique_ptr<scribblez::Agent> {
    return scribblez::make_player(players[seat], session.get(), players[1 - seat].display_name());
  };

  std::ofstream of;
  std::ostream* out = &std::cout;
  if (!out_path.empty()) {
    of.open(out_path);
    if (!of) {
      std::cerr << "Failed to open output file: " << out_path << "\n";
      return 1;
    }
    out = &of;
  }

  // Play `games` games (back to back, reusing the loaded dictionary), writing
  // each GCG to the output and tallying results.
  std::array<int, 2> wins = {0, 0};
  int draws = 0;
  long total_turns = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int gi = 0; gi < games; ++gi) {
    std::unique_ptr<scribblez::Agent> a0, a1;
    try {
      a0 = make_agent(0);
      a1 = make_agent(1);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 2;
    }
    scribblez::Game game(std::move(a0), std::move(a1), dict, seed + static_cast<uint64_t>(gi));
    game.play();
    const scribblez::GameLog& log = game.log();

    // Send the final position to the human (single-game human play only).
    if (human_seat >= 0 && session->connected()) {
      int opp = 1 - human_seat;
      scribblez::StateView final_view{game.board(),
                                      game.rack(human_seat),
                                      game.score(human_seat),
                                      game.score(opp),
                                      game.bag_size(),
                                      game.rack(opp).size(),
                                      players[human_seat].display_name(),
                                      players[opp].display_name(),
                                      /*legal_plays=*/nullptr,
                                      /*your_turn=*/false,
                                      /*game_over=*/true};
      session->send_text(scribblez::game_state_json(final_view));
      session->linger_after_final_message();
    }

    *out << scribblez::game_log_to_gcg(log);
    total_turns += static_cast<long>(log.turns.size());
    if (log.final_scores[0] > log.final_scores[1])
      ++wins[0];
    else if (log.final_scores[1] > log.final_scores[0])
      ++wins[1];
    else
      ++draws;

    if (verbose && games == 1) {
      std::cerr << "Final scores: " << log.final_scores[0] << " - " << log.final_scores[1] << "  ("
                << log.end_reason << ")\n";
      std::cerr << "Turns: " << log.turns.size() << "\n";
    }
  }

  if (verbose && games > 1) {
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << games << " games in " << secs << "s -> " << (games / secs) << " games/s, "
              << total_turns << " turns -> " << (total_turns / secs) << " moves/s\n";
    std::cerr << players[0].display_name() << " W/L/D vs " << players[1].display_name() << ": "
              << wins[0] << " / " << wins[1] << " / " << draws << "\n";
  }
  return 0;
}
