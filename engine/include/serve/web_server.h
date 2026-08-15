#pragma once

#include "agent/agent.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace boost::process {
class child;
class group;
}  // namespace boost::process

namespace scribblez {

class Game;

// Spawns `npm run dev` (the Vite dev server) to serve the front-end and
// terminates its whole process group on destruction. The browser loads the UI
// from Vite, which proxies the `/ws` WebSocket back to the WebSession. The
// engine launches the dev server itself, so no npm command is ever run by hand.
class ViteDevServer {
 public:
  // `dev_port` is the port Vite listens on and `ws_port` the WebSession port it
  // proxies `/ws` to; `tool` selects which front-end UI to mount, empty meaning
  // play_game's default. `service` and `default_dev_port` name this UI's
  // devenv.toml gateway service and its unshifted port, shaping only url().
  // Throws if the child process cannot be started.
  ViteDevServer(const std::string& web_dir, int dev_port, int ws_port, const std::string& tool,
                const std::string& service, int default_dev_port);
  ~ViteDevServer();

  ViteDevServer(const ViteDevServer&) = delete;
  ViteDevServer& operator=(const ViteDevServer&) = delete;

  // False on timeout or if the child exited early.
  bool wait_until_ready(int timeout_ms = 60000);

  // The gateway route for `service` while `dev_port` is the default, else a
  // plain localhost fallback (see service_url.h).
  std::string url() const;
  int dev_port() const { return dev_port_; }

 private:
  int dev_port_;
  int ws_port_;
  std::string tool_;
  std::string service_;
  int default_dev_port_;
  std::unique_ptr<boost::process::group> group_;
  std::unique_ptr<boost::process::child> child_;
};

// A minimal, single-client WebSocket server (POSIX sockets, blocking) that
// upgrades the `/ws` connection Vite proxies in and drives a human player. For
// local human-vs-AI play: one browser tab, one game, no concurrency.
class WebSession {
 public:
  // Throws util::CleanException if the port cannot be bound (another instance
  // holds it), util::Exception on other socket failures.
  explicit WebSession(int port);
  ~WebSession();

  WebSession(const WebSession&) = delete;
  WebSession& operator=(const WebSession&) = delete;

  // Accepts until a client completes the WebSocket handshake, closing
  // non-WebSocket requests; returns immediately if one is already connected.
  // False only on unrecoverable error.
  bool wait_for_client();

  // No-op if disconnected.
  void send_text(const std::string& msg);

  // nullopt once the connection closes. Control frames are handled
  // transparently.
  std::optional<std::string> recv_text();

  bool connected() const { return ws_fd_ >= 0; }
  void disconnect();
  int port() const { return port_; }

  // Keep the socket open briefly, so the final message lands before the process
  // exits.
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

// Build the GameState JSON the front-end expects, from the perspective of one
// seat ("my" side). Its two constructors cover the two sources a view is ever
// built from: mid-game, a MoveRequest (the human's own turn), whose fields it
// relays with the agent-supplied display names and optional Macondo equities
// tagged on; and end-of-game (or any other view anchored on a live Game), the
// Game itself.
//
// `legal_play_equities`, when non-null, must be parallel to `legal_plays`
// and is emitted per-move as the JSON `equity` field (null for entries
// without a value).
struct StateView {
  // The active-turn view, always your_turn and never game_over.
  // `display_moves` is the UI's "Legal Moves" panel -- legal plays plus any
  // synthesised exchanges -- and must outlive the StateView.
  StateView(const MoveRequest& req, const std::string& my_name, const std::string& opp_name,
            const std::vector<Move>& display_moves,
            const std::vector<std::optional<double>>* legal_play_equities = nullptr);

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
