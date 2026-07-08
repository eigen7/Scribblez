#pragma once

#include "agent/agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declarations so we can hold boost::process handles without dragging
// the (heavy) boost::process headers into every translation unit.
namespace boost::process {
class child;
class group;
}  // namespace boost::process

namespace scribblez {

// Forward decl so StateView can hold a Game-based ctor without web_server.h
// having to include game.h (game.h pulls in <vector>/<array>/etc. via the
// player log machinery and is not needed by anything else here).
class Game;

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
  // VITE_WS_PORT). `tool` selects which front-end UI to mount (passed through as
  // env var VITE_TOOL; empty means play_game's default UI). Throws if the child
  // process cannot be started.
  ViteDevServer(const std::string& web_dir, int dev_port, int ws_port, const std::string& tool);
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
  std::string tool_;
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

// Standard Scrabble coordinate notation for a play, e.g. "8H WAREZ 54"
// (horizontal) or "H8 WAREZ 54" (vertical). Newly placed blanks are lowercased.
// Non-plays render as "exch AQWW" (surrendered tiles, '?' for a blank) or
// "pass".
std::string move_to_notation(const Board& board, const Move& move);

// Build the GameState JSON the front-end expects, from the perspective of
// one seat ("my" side). Two constructors cover the only two places we ever
// build one of these:
//
//   * mid-game, from a MoveRequest (the human's own turn) -- the agent
//     already has every field of `req` parallel-named; we just relay them
//     and tag on the agent-supplied display names and optional Macondo
//     equities.
//   * end-of-game (or any other view derived from a live Game) -- pulled
//     directly from the Game itself, so callers no longer have to splat 11
//     positional fields into a brace-init list.
//
// `legal_play_equities`, when non-null, must be parallel to `legal_plays`
// and is emitted per-move as the JSON `equity` field (null for entries
// without a value).
struct StateView {
  // Active-turn ctor used by the human agent on every make_move() call. The
  // resulting view always has your_turn=true, game_over=false. `display_moves`
  // is the move list shown in the UI's "Legal Moves" panel (legal plays plus
  // any synthesised exchange moves); it must outlive the StateView.
  StateView(const MoveRequest& req, const std::string& my_name, const std::string& opp_name,
            const std::vector<Move>& display_moves,
            const std::vector<std::optional<double>>* legal_play_equities = nullptr);

  // Game-derived ctor used at game-over (or any other Game-anchored view).
  // `my_seat` selects which side is "me" (0 or 1).
  StateView(const Game& game, int my_seat, const std::string& my_name, const std::string& opp_name,
            bool your_turn, bool game_over);

  const Board& board;
  const Rack& my_rack;
  int my_score;
  int opp_score;
  int bag_size;
  int opp_rack_size;
  std::string my_name;
  std::string opp_name;
  const std::vector<Move>* legal_plays;
  const std::vector<std::optional<double>>* legal_play_equities;
  bool your_turn;
  bool game_over;
};
std::string game_state_json(const StateView& view);

}  // namespace scribblez
