// play_game: runs a single Scrabble game between two agents (Greedy or Human)
// and writes a JSON log. A Human player is driven through a local web UI.
//
// Usage:
//   play_game [--players P0,P1] [--kwg <path>] [--seed N]
//             [--out game.json] [--port N] [--web-dir DIR] [--verbose]
//   where each Pi is "greedy" or "human" (default: greedy,greedy).

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "scribblez/agent.h"
#include "scribblez/dictionary.h"
#include "scribblez/game.h"
#include "scribblez/json_writer.h"
#include "scribblez/web_server.h"

namespace {

// Default lexicon location, relative to the current working directory. The .kwg
// binary is not committed (it encodes a copyrighted wordlist); place or symlink
// it here, or point --kwg elsewhere.
constexpr const char* kDefaultKwg = "data/lexica/NWL23.kwg";
constexpr const char* kDefaultWebDir = "web/dist";

void usage() {
  std::cerr << "Usage: play_game [--players P0,P1] [--kwg <lexicon.kwg>] "
               "[--seed N] [--out game.json] [--port N] [--web-dir DIR] "
               "[--verbose]\n"
               "  --players: each of P0,P1 is 'greedy' or 'human' "
               "(default: greedy,greedy)\n"
               "  --kwg defaults to "
            << kDefaultKwg << "\n"
            << "  --port defaults to 8080, --web-dir to " << kDefaultWebDir
            << " (used only when a human plays)\n";
}

std::string to_lower(std::string s) {
  for (char& c : s) if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string kwg_path;
  std::string out_path;
  std::string web_dir = kDefaultWebDir;
  std::array<std::string, 2> player_types = {"greedy", "greedy"};
  int port = 8080;
  uint64_t seed = 0;
  bool seed_given = false;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing argument for " << flag << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--kwg" || a == "--dict") kwg_path = need(a);
    else if (a == "--out") out_path = need(a);
    else if (a == "--seed") { seed = std::stoull(need(a)); seed_given = true; }
    else if (a == "--port") port = std::stoi(need(a));
    else if (a == "--web-dir") web_dir = need(a);
    else if (a == "--players") {
      std::string spec = need(a);
      size_t comma = spec.find(',');
      if (comma == std::string::npos) {
        std::cerr << "--players expects two comma-separated types, e.g. "
                     "human,greedy\n";
        return 2;
      }
      player_types[0] = to_lower(spec.substr(0, comma));
      player_types[1] = to_lower(spec.substr(comma + 1));
    }
    else if (a == "--verbose" || a == "-v") verbose = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else {
      std::cerr << "Unknown argument: " << a << "\n";
      usage();
      return 2;
    }
  }

  // Validate player types and locate the human seat (at most one supported,
  // since a single browser drives the game).
  int human_seat = -1;
  for (int s = 0; s < 2; ++s) {
    if (player_types[s] != "greedy" && player_types[s] != "human") {
      std::cerr << "Unknown player type '" << player_types[s]
                << "' (expected 'greedy' or 'human')\n";
      return 2;
    }
    if (player_types[s] == "human") {
      if (human_seat >= 0) {
        std::cerr << "At most one human player is supported.\n";
        return 2;
      }
      human_seat = s;
    }
  }

  // Resolve the lexicon: explicit flag, else the default path.
  if (kwg_path.empty()) kwg_path = kDefaultKwg;
  if (!seed_given) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
  }

  scribblez::Dictionary dict;
  try {
    dict = scribblez::Dictionary::load_kwg(kwg_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n"
              << "No lexicon at '" << kwg_path << "'. Place or symlink an NWL "
                 ".kwg there, or pass --kwg <path>.\n";
    return 1;
  }
  if (verbose) {
    std::cerr << "Loaded KWG (" << dict.num_nodes() << " nodes) from " << kwg_path << "\n";
    std::cerr << "Seed: " << seed << "\n";
  }

  // Names: a human shows as "You", the AI as "Greedy".
  auto name_for = [](const std::string& type) {
    return type == "human" ? std::string("You") : std::string("Greedy");
  };

  // Stand up the web server up front (if a human is playing) so the browser can
  // connect before the game loop reaches the human's first turn.
  std::unique_ptr<scribblez::WebSession> session;
  if (human_seat >= 0) {
    try {
      session = std::make_unique<scribblez::WebSession>(port, web_dir);
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return 1;
    }
    std::cerr << "\n  Human-vs-AI game ready.\n"
              << "  Open  http://localhost:" << port << "  in your browser to play.\n\n";
    std::string cmd = "xdg-open http://localhost:" + std::to_string(port) +
                      " >/dev/null 2>&1 &";
    int rc = std::system(cmd.c_str());  // best-effort; ignore failure
    (void)rc;
  }

  auto make_agent = [&](int seat) -> std::unique_ptr<scribblez::Agent> {
    if (player_types[seat] == "human") {
      return std::make_unique<scribblez::HumanWebAgent>(
          *session, name_for(player_types[seat]),
          name_for(player_types[1 - seat]));
    }
    return std::make_unique<scribblez::GreedyAgent>();
  };

  scribblez::Game game(make_agent(0), make_agent(1), dict, seed);
  game.play();

  const auto& log = game.log();

  // Send the final position to the human and hold the connection briefly so the
  // game-over banner is delivered.
  if (human_seat >= 0 && session->connected()) {
    int opp = 1 - human_seat;
    scribblez::StateView final_view{game.board(),
                                    game.rack(human_seat),
                                    game.score(human_seat),
                                    game.score(opp),
                                    game.bag_size(),
                                    game.rack(opp).size(),
                                    name_for(player_types[human_seat]),
                                    name_for(player_types[opp]),
                                    /*legal_plays=*/nullptr,
                                    /*your_turn=*/false,
                                    /*game_over=*/true};
    session->send_text(scribblez::game_state_json(final_view));
    session->linger_after_final_message();
  }

  std::string json = scribblez::game_log_to_json(log);
  if (out_path.empty()) {
    std::cout << json;
  } else {
    std::ofstream of(out_path);
    if (!of) {
      std::cerr << "Failed to open output file: " << out_path << "\n";
      return 1;
    }
    of << json;
  }

  if (verbose) {
    std::cerr << "Final scores: " << log.final_scores[0] << " - "
              << log.final_scores[1] << "  (" << log.end_reason << ")\n";
    std::cerr << "Turns: " << log.turns.size() << "\n";
  }
  return 0;
}
