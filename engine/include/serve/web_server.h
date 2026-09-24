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

// Runs the Vite dev server (`npm run dev`) that serves the web front-end, and
// kills its whole process group on destruction. The browser loads the UI from
// Vite, which proxies the `/ws` WebSocket back to the engine's WebSession.
// Reuses a responsive dev server already on the port.
class ViteDevServer {
 public:
  // `dev_port` is the port Vite listens on and `ws_port` the WebSession port it
  // proxies `/ws` to. `tool` selects the front-end UI (web/src/main.tsx); empty
  // means play_game's. `service` and `default_dev_port` name the UI's
  // devenv.toml gateway service and its default port, and affect only url().
  // Throws if the child process cannot be started.
  ViteDevServer(const std::string& web_dir, int dev_port, int ws_port, const std::string& tool,
                const std::string& service, int default_dev_port);
  ~ViteDevServer();

  ViteDevServer(const ViteDevServer&) = delete;
  ViteDevServer& operator=(const ViteDevServer&) = delete;

  // Wait for the dev server to accept connections. Throws
  // util::CleanException, pointing at Vite's log, on timeout or if the child
  // exited.
  void wait_until_ready(int timeout_ms = 60000);

  // The browser-facing URL (service_url.h).
  std::string url() const;
  int dev_port() const { return dev_port_; }

 private:
  // False on timeout or if the child exited.
  bool ready_within(int timeout_ms);

  int dev_port_;
  int ws_port_;
  std::string tool_;
  std::string service_;
  int default_dev_port_;
  // <web_dir>/.vite-dev.log, where Vite's output goes so it cannot corrupt
  // play_game's stdout, which may carry the game-log JSON.
  std::string log_path_;
  std::unique_ptr<boost::process::group> group_;
  std::unique_ptr<boost::process::child> child_;
};

// A minimal, blocking, single-client WebSocket server on POSIX sockets. It
// accepts the `/ws` connection Vite proxies in and exchanges text messages with
// one browser tab: a human player in play_game, or an interactive tool UI.
// Binds loopback only. On construction it kills whatever process is listening
// on its port, so a relaunch takes over from an earlier instance.
class WebSession {
 public:
  // Throws util::CleanException if the port cannot be bound (another instance
  // holds it), util::Exception on other socket failures.
  explicit WebSession(int port);
  ~WebSession();

  WebSession(const WebSession&) = delete;
  WebSession& operator=(const WebSession&) = delete;

  // Block until a client completes the WebSocket handshake, closing any other
  // requests. Returns at once if a client is already connected. False only on
  // an unrecoverable error.
  bool wait_for_client();

  // No-op if disconnected.
  void send_text(const std::string& msg);

  // The next complete message, reassembled from fragments. nullopt once the
  // connection closes. Pings are answered internally.
  std::optional<std::string> recv_text();

  bool connected() const { return ws_fd_ >= 0; }
  void disconnect();
  int port() const { return port_; }

  // Sleep briefly so the final message is flushed before the process exits.
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

// The inputs to game_state_json(): the front-end's GameState, seen from one
// seat ("my" side). Built either from a MoveRequest, on the human's own turn,
// or from a live Game, e.g. at game end.
struct StateView {
  // The human's-turn view. `display_moves` fills the UI's move list (legal
  // plays plus any synthesized exchanges) and must outlive the view.
  // `legal_play_equities`, if non-null, runs parallel to it and supplies each
  // move's `equity` field (null for every move if it is null).
  StateView(const MoveRequest& req, const std::string& my_name, const std::string& opp_name,
            const std::vector<Move>& display_moves,
            const std::vector<double>* legal_play_equities = nullptr);

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
  const std::vector<double>* legal_play_equities;
  bool your_turn;
  bool game_over;
};
// The GameState JSON, plus the move list on the human's turn and the result
// once the game is over.
std::string game_state_json(const StateView& view);

}  // namespace scribblez
