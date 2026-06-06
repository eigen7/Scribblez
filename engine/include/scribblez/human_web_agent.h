#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class ViteDevServer;
class WebSession;

// A human player driven through a WebSession: renders the position to the
// browser and blocks until the user submits a move (or passes / exchanges).
// Owns the WebSession (the engine-side WebSocket server) and the Vite dev
// server that serves the front-end -- both are created in the constructor
// and torn down when the agent is destroyed, so the lifetime of the whole
// browser-driven UI is bound to the lifetime of the agent.
class HumanWebAgent : public Agent {
 public:
  // Per-instance configuration parsed from `--player "--type=human ..."`.
  // Defaults match what the top-level CLI used to provide.
  struct Params {
    int port = 8080;          // engine WebSocket port
    int vite_port = 5173;     // browser UI (Vite) port
    std::string web_dir = "web";  // front-end package dir (cwd of `npm run dev`)
  };

  // Constructs the engine-side WebSocket server, spawns `npm run dev` from
  // `params.web_dir`, blocks until Vite is ready, and best-effort opens the
  // browser at the Vite URL. Throws if any of those steps fail.
  HumanWebAgent(const Params& params, const std::string& my_name,
                const std::string& opp_name);
  ~HumanWebAgent() override;

  std::string name() const override { return my_name_; }
  Move make_move(const MoveRequest& req) override;

  // Surfaces the final board to the user and prompts them with Play Again /
  // Quit buttons; blocks until the browser responds, then returns the
  // corresponding EndGameAction.
  EndGameResult end_game(const Game& game, int my_seat) override;

  // Human players drive an interactive browser session and cannot safely run
  // concurrently with other games in a thread pool.
  bool supports_parallelism() const override { return false; }

  // Build a HumanWebAgent from `--player "--type=human [options]"` tokens
  // (after the factory has stripped --type and --name). Throws on bad input.
  static std::unique_ptr<HumanWebAgent> from_spec(const std::vector<std::string>& tokens,
                                                  const std::string& name,
                                                  const std::string& opp_name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  std::unique_ptr<WebSession> session_;
  std::unique_ptr<ViteDevServer> vite_;
  std::string my_name_;
  std::string opp_name_;
};

}  // namespace scribblez
