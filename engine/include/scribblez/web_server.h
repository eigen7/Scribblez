#pragma once
#include <optional>
#include <string>
#include <vector>

#include "scribblez/agent.h"
#include "scribblez/board.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

namespace scribblez {

// A minimal, single-client HTTP + WebSocket server (POSIX sockets, blocking).
// It serves the built web UI as static files and upgrades GET /ws to a
// WebSocket that drives a human player. Designed for local human-vs-AI play:
// exactly one browser tab, one game, no concurrency.
class WebSession {
 public:
  // Bind and listen on `port`, serving static files from `web_dir`. Throws
  // std::runtime_error on failure.
  WebSession(int port, std::string web_dir);
  ~WebSession();

  WebSession(const WebSession&) = delete;
  WebSession& operator=(const WebSession&) = delete;

  // Block accepting connections -- serving any static-file requests inline --
  // until a browser completes the WebSocket handshake. Returns immediately if a
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
  void serve_static(int fd, const std::string& path);
  bool do_handshake(int fd, const std::string& request);

  int port_;
  std::string web_dir_;
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
