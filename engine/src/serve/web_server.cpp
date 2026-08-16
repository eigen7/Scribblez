#include "serve/web_server.h"

#include "game/game.h"
#include "game/tile.h"
#include "serve/position_json.h"
#include "serve/service_url.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/string.h"

#include <arpa/inet.h>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/filesystem.hpp>
#include <boost/json.hpp>
#include <boost/process.hpp>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <format>
#include <iostream>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace scribblez {

namespace {

// Generic socket I/O, Base64, SHA-1, and ASCII case folding live in util/.
using util::base64;
using util::read_n;
using util::sha1;
using util::to_lower;
using util::write_all;

// --------------------------- HTTP / WS helpers ---------------------------

// Case-insensitive.
std::string header_value(const std::string& req, const std::string& name) {
  std::string lower = req;
  to_lower(lower);
  std::string key = name;
  to_lower(key);
  key += ":";
  size_t pos = lower.find(key);
  if (pos == std::string::npos) return "";
  pos += key.size();
  size_t end = req.find("\r\n", pos);
  std::string val = req.substr(pos, end - pos);
  size_t a = val.find_first_not_of(" \t");
  size_t b = val.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : val.substr(a, b - a + 1);
}

// --------------------------- JSON serialization --------------------------

// ----------------------------- port freeing ------------------------------

// PIDs currently LISTENing on `port`, via lsof. We intentionally ignore
// established client sockets so we never kill unrelated client processes.
std::set<int> pids_listening_on_port(int port) {
  namespace bp = boost::process;
  std::set<int> pids;
  boost::filesystem::path lsof = bp::search_path("lsof");
  if (lsof.empty()) return pids;  // can't probe; treat as free

  bp::ipstream out;
  std::error_code ec;
  bp::child c(lsof, "-nP", "-t", std::format("-iTCP:{}", port), "-sTCP:LISTEN", bp::std_out > out,
              bp::std_err > bp::null, ec);
  if (ec) return pids;
  std::string line;
  while (out && std::getline(out, line)) {
    boost::trim(line);
    if (!line.empty()) {
      try {
        pids.insert(std::stoi(line));
      } catch (const std::exception&) {
      }
    }
  }
  c.wait();
  return pids;
}

bool http_responding_on_port(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return false;

  timeval tv{};
  tv.tv_sec = 0;
  tv.tv_usec = 300000;  // 300ms
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(uint16_t(port));

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return false;
  }

  const char req[] = "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
  if (!write_all(fd, req, sizeof(req) - 1)) {
    ::close(fd);
    return false;
  }

  char buf[16];
  const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
  ::close(fd);
  return n > 0;
}

void kill_listening_pids_on_port(int port) {
  std::set<int> pids = pids_listening_on_port(port);
  pid_t self = ::getpid();
  bool killed_any = false;
  for (int pid : pids) {
    if (pid == self) continue;
    std::cerr << "  Port " << port << " is in use by pid " << pid << "; reclaiming it.\n";
    ::kill(pid, SIGKILL);
    killed_any = true;
  }
  if (!killed_any) return;

  for (int i = 0; i < 20; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::set<int> still = pids_listening_on_port(port);
    still.erase(self);
    if (still.empty()) return;
  }
}

}  // namespace

// --------------------------- move notation -------------------------------

std::string move_to_notation(const Board& board, const Move& move) {
  if (move.type() == MoveType::PASS) return "pass";
  if (move.type() == MoveType::EXCHANGE) {
    // The surrendered tiles are stored sorted (A..Z then blanks); render blanks
    // as '?'. e.g. "exch AQWW".
    std::string letters;
    for (int i = 0; i < move.num_glyphs(); ++i) {
      Glyph g = move.glyph(i);
      letters.push_back(g.is_blank() ? '?' : g.letter().to_char());
    }
    return "exch " + letters;
  }

  auto [sr, sc] = move.word_origin(board);
  std::string pos;
  char col_letter = char('A' + sc);
  if (move.horizontal()) {
    pos = std::format("{}{}", sr + 1, col_letter);  // e.g. "8H"
  } else {
    pos = std::format("{}{}", col_letter, sr + 1);  // e.g. "H8"
  }

  // Render the main word, lowercasing blank tiles (new or already on board).
  std::string word;
  const int dr = move.horizontal() ? 0 : 1, dc = move.horizontal() ? 1 : 0;
  const int n = move.num_glyphs();
  int r = sr, c = sc, gi = 0;
  while (board.in_bounds(r, c)) {
    Glyph cell = board.at(r, c);
    Glyph g;
    if (!cell.is_empty()) {
      g = cell;  // existing tile
    } else if (gi < n) {
      g = move.glyph(gi++);  // newly placed tile
    } else {
      break;
    }
    char ch = g.letter().to_char();
    word.push_back(g.is_blank() ? char(ch - 'A' + 'a') : ch);
    r += dr;
    c += dc;
  }

  return std::format("{} {} {}", pos, word, move.score());
}

// --------------------------- state serialization -------------------------

StateView::StateView(const MoveRequest& req, const std::string& my_name,
                     const std::string& opp_name, const std::vector<Move>& display_moves,
                     const std::vector<std::optional<double>>* equities)
    : board(req.board),
      my_rack(req.my_rack),
      my_score(req.my_score),
      opp_score(req.opp_score),
      bag_size(req.bag_size),
      opp_rack_size(req.opp_rack.size()),
      my_name(my_name),
      opp_name(opp_name),
      legal_plays(&display_moves),
      legal_play_equities(equities),
      your_turn(true),
      game_over(false) {}

StateView::StateView(const Game& game, int my_seat, const std::string& my_name,
                     const std::string& opp_name, bool your_turn, bool game_over)
    : board(game.board()),
      my_rack(game.rack(my_seat)),
      my_score(game.score(my_seat)),
      opp_score(game.score(1 - my_seat)),
      bag_size(game.bag_size()),
      opp_rack_size(game.rack(1 - my_seat).size()),
      my_name(my_name),
      opp_name(opp_name),
      legal_plays(nullptr),
      legal_play_equities(nullptr),
      your_turn(your_turn),
      game_over(game_over) {}

std::string game_state_json(const StateView& v) {
  namespace json = boost::json;
  const bool game_over = v.game_over;

  // Common GameState fields (board, bonuses, rack, scores, counts, tile_scores)
  // come from the shared serializer; this view appends the live move list and
  // game-over result.
  json::object o =
    position_state_object(v.board, v.my_rack, v.my_score, v.opp_score, v.bag_size, v.opp_rack_size,
                          v.my_name, v.opp_name, v.your_turn, game_over);

  // moves: only on the human's live turn.
  if (v.legal_plays && v.your_turn && !game_over) {
    json::array moves;
    for (size_t i = 0; i < v.legal_plays->size(); ++i) {
      const Move& m = (*v.legal_plays)[i];
      json::object mo{
        {"index", int(i)}, {"text", move_to_notation(v.board, m)}, {"score", m.score()}};
      // equity: null when we have no Macondo evaluation (or no value for
      // this particular play). The front-end renders the null cells blank.
      if (v.legal_play_equities && i < v.legal_play_equities->size() &&
          (*v.legal_play_equities)[i].has_value()) {
        mo["equity"] = (*v.legal_play_equities)[i].value();
      } else {
        mo["equity"] = nullptr;
      }
      moves.emplace_back(std::move(mo));
    }
    o["moves"] = std::move(moves);
  }

  if (game_over) {
    int winner = v.my_score > v.opp_score ? 0 : (v.opp_score > v.my_score ? 1 : -1);
    o["winner"] = winner;
    o["final_scores"] = {v.my_score, v.opp_score};
  }

  return json::serialize(o);
}

// ------------------------------ ViteDevServer ----------------------------

ViteDevServer::ViteDevServer(const std::string& web_dir, int dev_port, int ws_port,
                             const std::string& tool, const std::string& service,
                             int default_dev_port)
    : dev_port_(dev_port),
      ws_port_(ws_port),
      tool_(tool),
      service_(service),
      default_dev_port_(default_dev_port) {
  namespace bp = boost::process;

  // If a dev server is already listening on this port, reuse it only when it
  // actually responds to HTTP. If the listener is stale/stuck, reclaim it.
  if (!pids_listening_on_port(dev_port_).empty()) {
    if (http_responding_on_port(dev_port_)) {
      std::cerr << "  Reusing existing dev server on port " << dev_port_ << ".\n";
      return;
    }
    std::cerr << "  Existing listener on port " << dev_port_
              << " is unresponsive; restarting dev server.\n";
    kill_listening_pids_on_port(dev_port_);
  }

  // Pass the ports through to vite.config.ts so the browser UI and the engine's
  // WebSocket server always agree on which ports to use / proxy, and the tool
  // name through to main.tsx so the bare URL mounts the right UI.
  bp::environment env = boost::this_process::environment();
  env["VITE_DEV_PORT"] = std::to_string(dev_port_);
  env["VITE_WS_PORT"] = std::to_string(ws_port_);
  if (!tool_.empty()) env["VITE_TOOL"] = tool_;

  // Redirect Vite's chatty output to a log file so it can never corrupt
  // play_game's own stdout (which may carry the game-log JSON).
  boost::filesystem::path log = boost::filesystem::path(web_dir) / ".vite-dev.log";

  boost::filesystem::path npm = bp::search_path("npm");
  if (npm.empty()) {
    throw util::CleanException(
      "npm not found on PATH (needed to launch the web UI); run py/build.py");
  }

  // A process group lets us terminate the whole tree on exit: `npm run dev`
  // spawns vite as a grandchild that would otherwise be orphaned.
  group_ = std::make_unique<bp::group>();
  child_ = std::make_unique<bp::child>(npm, "run", "dev", bp::start_dir = web_dir,
                                       bp::std_out > log, bp::std_err > log, env, *group_);
}

ViteDevServer::~ViteDevServer() {
  boost::system::error_code ec;
  if (group_) group_->terminate(ec);  // kill the whole process group
  if (child_) child_->wait(ec);       // reap
}

bool ViteDevServer::wait_until_ready(int timeout_ms) {
  if (!child_) return http_responding_on_port(dev_port_);

  namespace asio = boost::asio;
  using tcp = asio::ip::tcp;

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  tcp::endpoint endpoint(asio::ip::make_address("127.0.0.1"), uint16_t(dev_port_));
  while (std::chrono::steady_clock::now() < deadline) {
    // Fail fast if the dev server already exited (e.g. deps not installed).
    if (child_ && !child_->running()) return false;

    asio::io_context io;
    tcp::socket sock(io);
    boost::system::error_code ec;
    if (!sock.connect(endpoint, ec)) return true;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return false;
}

std::string ViteDevServer::url() const {
  return service_url(service_, dev_port_, default_dev_port_);
}

// ------------------------------ WebSession -------------------------------

WebSession::WebSession(int port) : port_(port) {
  std::signal(SIGPIPE, SIG_IGN);
  // Reclaim the WebSocket port so a new manual_gcg_tool launch can take over
  // from an older instance without manual cleanup.
  kill_listening_pids_on_port(port_);
  // SOCK_CLOEXEC so the Vite dev server we later fork/exec does not inherit (and
  // keep alive) this listening socket.
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) throw util::Exception("socket() failed");
  int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(uint16_t(port_));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(listen_fd_);
    throw util::CleanException("bind() failed on port {} (is another instance running?)", port_);
  }
  if (::listen(listen_fd_, 16) < 0) {
    ::close(listen_fd_);
    throw util::Exception("listen() failed");
  }
}

WebSession::~WebSession() {
  if (ws_fd_ >= 0) ::close(ws_fd_);
  if (listen_fd_ >= 0) ::close(listen_fd_);
}

void WebSession::disconnect() {
  if (ws_fd_ >= 0) {
    ::close(ws_fd_);
    ws_fd_ = -1;
  }
}

bool WebSession::do_handshake(int fd, const std::string& request) {
  std::string key = header_value(request, "Sec-WebSocket-Key");
  if (key.empty()) return false;
  uint8_t digest[20];
  sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", digest);
  std::string accept = base64(digest, 20);
  std::string resp =
    "HTTP/1.1 101 Switching Protocols\r\n"
    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Accept: " +
    accept + "\r\n\r\n";
  return write_all(fd, resp.data(), resp.size());
}

bool WebSession::wait_for_client() {
  if (ws_fd_ >= 0) return true;
  for (;;) {
    int conn = ::accept(listen_fd_, nullptr, nullptr);
    if (conn < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    // Read the request headers, which a blank line terminates.
    std::string req;
    char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos) {
      ssize_t r = ::recv(conn, buf, sizeof(buf), 0);
      if (r <= 0) break;
      req.append(buf, size_t(r));
      if (req.size() > 64 * 1024) break;
    }
    if (req.empty()) {
      ::close(conn);
      continue;
    }

    std::string upgrade = header_value(req, "Upgrade");
    to_lower(upgrade);
    if (upgrade == "websocket") {
      if (do_handshake(conn, req)) {
        ws_fd_ = conn;
        return true;
      }
      ::close(conn);
      continue;
    }
    // Not a WebSocket upgrade. The UI is served by the Vite dev server, which
    // only proxies `/ws` here, so any other request is stray -- just close it.
    ::close(conn);
  }
}

void WebSession::send_text(const std::string& msg) {
  if (ws_fd_ < 0) return;
  std::string frame;
  frame.push_back(char(0x81));  // FIN + text
  size_t n = msg.size();
  if (n < 126) {
    frame.push_back(char(n));
  } else if (n < 65536) {
    frame.push_back(char(126));
    frame.push_back(char((n >> 8) & 0xff));
    frame.push_back(char(n & 0xff));
  } else {
    frame.push_back(char(127));
    for (int i = 7; i >= 0; --i) frame.push_back(char((uint64_t(n) >> (i * 8)) & 0xff));
  }
  frame += msg;
  if (!write_all(ws_fd_, frame.data(), frame.size())) disconnect();
}

std::optional<std::string> WebSession::recv_text() {
  if (ws_fd_ < 0) return std::nullopt;
  // Accumulates the payload of a fragmented message: a data frame with FIN
  // clear, followed by continuation frames (opcode 0x0) until one has FIN set.
  // Browsers fragment large messages (e.g. a big PNG export), so we must
  // reassemble rather than return the first frame.
  std::string message;
  for (;;) {
    uint8_t hdr[2];
    if (!read_n(ws_fd_, hdr, 2)) {
      disconnect();
      return std::nullopt;
    }
    bool fin = (hdr[0] & 0x80) != 0;
    int opcode = hdr[0] & 0x0f;
    bool masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7f;
    if (len == 126) {
      uint8_t ext[2];
      if (!read_n(ws_fd_, ext, 2)) {
        disconnect();
        return std::nullopt;
      }
      len = (uint64_t(ext[0]) << 8) | ext[1];
    } else if (len == 127) {
      uint8_t ext[8];
      if (!read_n(ws_fd_, ext, 8)) {
        disconnect();
        return std::nullopt;
      }
      len = 0;
      for (int i = 0; i < 8; ++i) len = (len << 8) | ext[i];
    }
    uint8_t mask[4] = {0, 0, 0, 0};
    if (masked && !read_n(ws_fd_, mask, 4)) {
      disconnect();
      return std::nullopt;
    }
    std::string payload(len, '\0');
    if (len && !read_n(ws_fd_, payload.data(), len)) {
      disconnect();
      return std::nullopt;
    }
    if (masked)
      for (uint64_t i = 0; i < len; ++i) payload[i] ^= mask[i & 3];

    switch (opcode) {
      case 0x0:  // continuation
      case 0x1:  // text
      case 0x2:  // binary
        message += payload;
        if (fin) return message;
        break;   // more fragments to come
      case 0x8:  // close
        disconnect();
        return std::nullopt;
      case 0x9: {  // ping -> pong
        std::string frame;
        frame.push_back(char(0x8a));
        frame.push_back(char(payload.size() & 0x7f));
        frame += payload;
        if (!write_all(ws_fd_, frame.data(), frame.size())) {
          disconnect();
          return std::nullopt;
        }
        break;
      }
      default:
        break;  // pong: ignore
    }
  }
}

void WebSession::linger_after_final_message() {
  // Give the kernel a moment to flush the final frame before we close.
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
}

}  // namespace scribblez
