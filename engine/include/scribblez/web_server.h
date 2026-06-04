#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "scribblez/agent.h"
#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

// Forward declarations so we can hold boost::process handles without dragging
// the (heavy) boost::process headers into every translation unit.
namespace boost::process {
class child;
class group;
}  // namespace boost::process

namespace scribblez {

// Spawns `npm run dev` (the Vite dev server) as a child process to serve the
// front-end, and terminates it (and its whole process group) on destruction.
// The browser loads the UI from Vite, which proxies the `/ws` WebSocket back to
// the WebSession. The build step installs the npm deps, and the engine launches
// the dev server itself, so no npm commands are ever run by hand.
class ViteDevServer {
 public:
  // Launch `npm run dev` with cwd `web_dir` (the front-end package directory).
  // `dev_port` is the port Vite listens on; `ws_port` is the WebSession port
  // Vite proxies `/ws` to (passed through as env vars VITE_DEV_PORT /
  // VITE_WS_PORT). Throws if the child process cannot be started.
  ViteDevServer(std::string web_dir, int dev_port, int ws_port);
  ~ViteDevServer();

  ViteDevServer(const ViteDevServer&) = delete;
  ViteDevServer& operator=(const ViteDevServer&) = delete;

  // Block until Vite is accepting connections, or until `timeout_ms` elapses.
  // Returns false on timeout or if the child exited early.
  bool wait_until_ready(int timeout_ms = 60000);

  // URL to open in the browser, e.g. "http://localhost:5173".
  std::string url() const;
  int dev_port() const { return dev_port_; }

 private:
  int dev_port_;
  int ws_port_;
  std::unique_ptr<boost::process::group> group_;
  std::unique_ptr<boost::process::child> child_;
};

// A minimal, single-client WebSocket server (POSIX sockets, blocking). It
// upgrades the `/ws` connection (proxied in by the Vite dev server) to a
// WebSocket that drives a human player. Designed for local human-vs-AI play:
// exactly one browser tab, one game, no concurrency.
class WebSession {
 public:
  // Bind and listen on `port`. Throws std::runtime_error on failure.
  explicit WebSession(int port);
  ~WebSession();

  WebSession(const WebSession&) = delete;
  WebSession& operator=(const WebSession&) = delete;

  // Block accepting connections until a client completes the WebSocket
  // handshake (non-WebSocket requests are closed). Returns immediately if a
  // client is already connected. Returns false only on unrecoverable error.
  bool wait_for_client();

  // Send one text (JSON) message to the connected client. No-op if disconnected.
  void send_text(const std::string& msg);

  // Receive one text message from the client. Returns std::nullopt if the
  // connection closed (control frames are handled transparently).
  std::optional<std::string> recv_text();

  bool connected() const { return ws_fd_ >= 0; }
  void disconnect();
  int port() const { return port_; }

  // After the game ends, keep the socket open briefly so the final message is
  // delivered before the process exits.
  void linger_after_final_message();

 private:
  bool do_handshake(int fd, const std::string& request);

  int port_;
  int listen_fd_ = -1;
  int ws_fd_ = -1;
};

// A human player driven through a WebSession: renders the position to the
// browser and blocks until the user submits a move (or passes / exchanges).
class HumanWebAgent : public Agent {
 public:
  HumanWebAgent(WebSession& session, std::string my_name, std::string opp_name);
  std::string name() const override { return my_name_; }
  Move choose(const AgentContext& ctx, std::mt19937_64& rng) override;

 private:
  WebSession& session_;
  std::string my_name_;
  std::string opp_name_;
};

// Standard Scrabble coordinate notation for a play, e.g. "8H WAREZ 54"
// (horizontal) or "H8 WAREZ 54" (vertical). Newly placed blanks are lowercased.
std::string move_to_notation(const Board& board, const Move& move);

// Build the GameState JSON the front-end expects, from the human's perspective
// (index 0 is always the human). `legal_plays` is rendered as the selectable
// move list only when `your_turn` is true.
struct StateView {
  const Board& board;
  const Rack& my_rack;
  int my_score = 0;
  int opp_score = 0;
  int bag_size = 0;
  int opp_rack_size = 0;
  std::string my_name;
  std::string opp_name;
  const std::vector<Move>* legal_plays = nullptr;
  bool your_turn = false;
  bool game_over = false;
};
std::string game_state_json(const StateView& view);

}  // namespace scribblez
