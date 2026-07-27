#pragma once

#include "agent/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class ViteDevServer;
class WebSession;

// A human player driven through a WebSession: renders the position to the
// browser and blocks until the user submits a move. Owns both the engine-side
// WebSocket server and the Vite dev server serving the front-end, so the whole
// browser-driven UI lives and dies with the agent.
class HumanWebAgent : public Agent {
 public:
  // The browser link is routed through the gateway only while vite_port keeps
  // its default (see service_url.h).
  struct Params {
    int port = 8080;              // engine WebSocket port
    int vite_port = 5173;         // browser UI (Vite) port
    std::string web_dir = "web";  // front-end package dir (cwd of `npm run dev`)
  };

  // Starts the WebSocket server, spawns `npm run dev`, blocks until Vite is
  // ready, and best-effort opens the browser. Throws if any step fails.
  HumanWebAgent(int thread_id, const Params& params, const std::string& my_name,
                const std::string& opp_name);
  ~HumanWebAgent() override;

  MoveDecision make_move(const MoveRequest& req) override;

  // Surfaces the final board and blocks on the user's Play Again / Quit choice.
  EndGameResult end_game(const Game& game, int my_seat) override;

  bool supports_parallelism() const override { return false; }

  // Build from `--player "--type=human [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<HumanWebAgent> from_spec(const std::vector<std::string>& tokens,
                                                  int thread_id, const std::string& name,
                                                  const std::string& opp_name);

  static std::string options_help();

 private:
  std::unique_ptr<WebSession> session_;
  std::unique_ptr<ViteDevServer> vite_;
  std::string opp_name_;
};

}  // namespace scribblez
