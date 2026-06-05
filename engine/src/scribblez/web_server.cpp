#include "scribblez/web_server.h"

#include "scribblez/tile.h"

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
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>

namespace scribblez {

namespace {

// ----------------------------- low-level I/O -----------------------------

// Read exactly n bytes into buf. Returns false on EOF / error.
bool read_n(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd, p + got, n - got, 0);
    if (r <= 0) return false;
    got += static_cast<size_t>(r);
  }
  return true;
}

bool write_all(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < n) {
    ssize_t w = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
    if (w <= 0) return false;
    sent += static_cast<size_t>(w);
  }
  return true;
}

// ------------------------------- SHA-1 -----------------------------------

void sha1(const std::string& msg, uint8_t out[20]) {
  uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
  std::string data = msg;
  uint64_t bits = static_cast<uint64_t>(data.size()) * 8;
  data.push_back(static_cast<char>(0x80));
  while (data.size() % 64 != 56) data.push_back(0);
  for (int i = 7; i >= 0; --i) data.push_back(static_cast<char>((bits >> (i * 8)) & 0xff));

  auto rol = [](uint32_t v, int b) { return (v << b) | (v >> (32 - b)); };
  for (size_t chunk = 0; chunk < data.size(); chunk += 64) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint8_t>(data[chunk + i * 4]) << 24) |
             (static_cast<uint8_t>(data[chunk + i * 4 + 1]) << 16) |
             (static_cast<uint8_t>(data[chunk + i * 4 + 2]) << 8) |
             (static_cast<uint8_t>(data[chunk + i * 4 + 3]));
    }
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999u;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1u;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDCu;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6u;
      }
      uint32_t tmp = rol(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rol(b, 30);
      b = a;
      a = tmp;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
  }
  for (int i = 0; i < 5; ++i) {
    out[i * 4] = (h[i] >> 24) & 0xff;
    out[i * 4 + 1] = (h[i] >> 16) & 0xff;
    out[i * 4 + 2] = (h[i] >> 8) & 0xff;
    out[i * 4 + 3] = h[i] & 0xff;
  }
}

std::string base64(const uint8_t* data, size_t len) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = data[i] << 16;
    if (i + 1 < len) n |= data[i + 1] << 8;
    if (i + 2 < len) n |= data[i + 2];
    out.push_back(tbl[(n >> 18) & 63]);
    out.push_back(tbl[(n >> 12) & 63]);
    out.push_back(i + 1 < len ? tbl[(n >> 6) & 63] : '=');
    out.push_back(i + 2 < len ? tbl[n & 63] : '=');
  }
  return out;
}

// --------------------------- HTTP / WS helpers ---------------------------

std::string& to_lower(std::string& s) {
  for (char& c : s)
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

// Extract a header value (case-insensitive name) from a raw HTTP request.
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

const char* premium_code(Premium p) {
  switch (p) {
    case Premium::DLS:
      return "DL";
    case Premium::TLS:
      return "TL";
    case Premium::DWS:
      return "DW";
    case Premium::TWS:
      return "TW";
    default:
      return nullptr;
  }
}

// ----------------------------- port freeing ------------------------------

// PIDs currently listening on / connected to `port`, via `lsof -t -i :port`.
std::set<int> pids_on_port(int port) {
  namespace bp = boost::process;
  std::set<int> pids;
  boost::filesystem::path lsof = bp::search_path("lsof");
  if (lsof.empty()) return pids;  // can't probe; treat as free

  bp::ipstream out;
  std::error_code ec;
  bp::child c(lsof, "-t", "-i", ":" + std::to_string(port), bp::std_out > out,
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

// Kill any process holding `port` and wait for the OS to release it, so we can
// (re)bind / relaunch cleanly even after a previous run was killed abruptly and
// orphaned its child processes. Best-effort: silently no-op if lsof is absent.
void free_port(int port) {
  std::set<int> pids = pids_on_port(port);
  pid_t self = ::getpid();
  bool killed_any = false;
  for (int pid : pids) {
    if (pid == self) continue;
    std::cerr << "  Port " << port << " is in use by pid " << pid << "; freeing it.\n";
    ::kill(pid, SIGKILL);
    killed_any = true;
  }
  if (!killed_any) return;

  for (int i = 0; i < 10; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::set<int> still = pids_on_port(port);
    still.erase(self);
    if (still.empty()) return;
  }
}

}  // namespace

// --------------------------- move notation -------------------------------

std::string move_to_notation(const Board& board, const Move& move) {
  const int sr = move.start_row, sc = move.start_col;
  std::string pos;
  char col_letter = static_cast<char>('A' + sc);
  std::string row_num = std::to_string(sr + 1);
  if (move.horizontal) {
    pos = row_num + col_letter;  // e.g. "8H"
  } else {
    pos = col_letter + row_num;  // e.g. "H8"
  }

  // Render the main word, lowercasing blank tiles (new or already on board).
  std::string word;
  const int dr = move.horizontal ? 0 : 1, dc = move.horizontal ? 1 : 0;
  const int n = move.num_glyphs();
  int r = sr, c = sc, gi = 0;
  while (board.in_bounds(r, c)) {
    Glyph cell = board.at(r, c);
    Glyph g;
    if (!cell.is_empty()) {
      g = cell;  // existing tile
    } else if (gi < n) {
      g = move.glyphs[gi++];  // newly placed tile
    } else {
      break;
    }
    char ch = g.letter().to_char();
    word.push_back(g.is_blank() ? static_cast<char>(ch - 'A' + 'a') : ch);
    r += dr;
    c += dc;
  }

  return pos + " " + word + " " + std::to_string(move.score);
}

// --------------------------- state serialization -------------------------

std::string game_state_json(const StateView& v) {
  namespace json = boost::json;
  const bool game_over = v.game_over;

  json::object o;
  o["type"] = game_over ? "game_over" : "state";

  // board: 15x15, uppercase letters, lowercase for blanks, null for empty.
  json::array board;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph sq = v.board.at(r, c);
      if (sq.is_empty()) {
        row.emplace_back(nullptr);
      } else {
        char ch = sq.letter().to_char();
        if (sq.is_blank()) ch = ch - 'A' + 'a';
        row.emplace_back(std::string(1, ch));
      }
    }
    board.emplace_back(std::move(row));
  }
  o["board"] = std::move(board);

  // bonuses: DL/TL/DW/TW or null.
  json::array bonuses;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const char* code = premium_code(v.board.premium_at(r, c));
      if (code)
        row.emplace_back(code);
      else
        row.emplace_back(nullptr);
    }
    bonuses.emplace_back(std::move(row));
  }
  o["bonuses"] = std::move(bonuses);

  // rack: letters then blanks ('?'), with per-tile score.
  json::array rack;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    for (int i = 0; i < v.my_rack.count(L); ++i) {
      rack.emplace_back(
        json::object{{"letter", std::string(1, L.to_char())}, {"score", TILE_VALUES[L]}});
    }
  }
  for (int i = 0; i < v.my_rack.blanks(); ++i) {
    rack.emplace_back(json::object{{"letter", "?"}, {"score", 0}});
  }
  o["rack"] = std::move(rack);

  o["scores"] = {v.my_score, v.opp_score};
  o["player_names"] = {v.my_name, v.opp_name};
  o["bag_count"] = v.bag_size;
  o["opponent_rack_count"] = v.opp_rack_size;
  o["your_turn"] = v.your_turn;
  o["game_over"] = game_over;

  // tile_scores map: { "A": 1, "B": 3, ... }.
  json::object tile_scores;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    tile_scores[std::string(1, L.to_char())] = TILE_VALUES[L];
  }
  o["tile_scores"] = std::move(tile_scores);

  // moves: only on the human's live turn.
  if (v.legal_plays && v.your_turn && !game_over) {
    json::array moves;
    for (size_t i = 0; i < v.legal_plays->size(); ++i) {
      const Move& m = (*v.legal_plays)[i];
      json::object mo{{"index", static_cast<int>(i)},
                      {"text", move_to_notation(v.board, m)},
                      {"score", m.score}};
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

ViteDevServer::ViteDevServer(const std::string& web_dir, int dev_port, int ws_port)
    : dev_port_(dev_port), ws_port_(ws_port) {
  namespace bp = boost::process;

  // Reclaim the dev-server port in case a previous run's Vite was orphaned.
  free_port(dev_port_);

  // Pass the ports through to vite.config.ts so the browser UI and the engine's
  // WebSocket server always agree on which ports to use / proxy.
  bp::environment env = boost::this_process::environment();
  env["VITE_DEV_PORT"] = std::to_string(dev_port_);
  env["VITE_WS_PORT"] = std::to_string(ws_port_);

  // Redirect Vite's chatty output to a log file so it can never corrupt
  // play_game's own stdout (which may carry the game-log JSON).
  boost::filesystem::path log = boost::filesystem::path(web_dir) / ".vite-dev.log";

  boost::filesystem::path npm = bp::search_path("npm");
  if (npm.empty()) {
    throw std::runtime_error("npm not found on PATH (needed to launch the web UI); run ./build.py");
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
  namespace asio = boost::asio;
  using tcp = asio::ip::tcp;

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  tcp::endpoint endpoint(asio::ip::make_address("127.0.0.1"),
                         static_cast<unsigned short>(dev_port_));
  while (std::chrono::steady_clock::now() < deadline) {
    // Fail fast if the dev server already exited (e.g. deps not installed).
    if (child_ && !child_->running()) return false;

    asio::io_context io;
    tcp::socket sock(io);
    boost::system::error_code ec;
    static_cast<void>(sock.connect(endpoint, ec));
    if (!ec) return true;

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return false;
}

std::string ViteDevServer::url() const { return "http://localhost:" + std::to_string(dev_port_); }

// ------------------------------ WebSession -------------------------------

WebSession::WebSession(int port) : port_(port) {
  std::signal(SIGPIPE, SIG_IGN);
  // Reclaim the port if a previous run left something holding it.
  free_port(port_);
  // SOCK_CLOEXEC so the Vite dev server we later fork/exec does not inherit (and
  // keep alive) this listening socket.
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
  int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(listen_fd_);
    throw std::runtime_error("bind() failed on port " + std::to_string(port_) +
                             " (is another instance running?)");
  }
  if (::listen(listen_fd_, 16) < 0) {
    ::close(listen_fd_);
    throw std::runtime_error("listen() failed");
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
    // Read the request headers (terminated by a blank line).
    std::string req;
    char buf[2048];
    while (req.find("\r\n\r\n") == std::string::npos) {
      ssize_t r = ::recv(conn, buf, sizeof(buf), 0);
      if (r <= 0) break;
      req.append(buf, static_cast<size_t>(r));
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
  frame.push_back(static_cast<char>(0x81));  // FIN + text
  size_t n = msg.size();
  if (n < 126) {
    frame.push_back(static_cast<char>(n));
  } else if (n < 65536) {
    frame.push_back(static_cast<char>(126));
    frame.push_back(static_cast<char>((n >> 8) & 0xff));
    frame.push_back(static_cast<char>(n & 0xff));
  } else {
    frame.push_back(static_cast<char>(127));
    for (int i = 7; i >= 0; --i)
      frame.push_back(static_cast<char>((static_cast<uint64_t>(n) >> (i * 8)) & 0xff));
  }
  frame += msg;
  if (!write_all(ws_fd_, frame.data(), frame.size())) disconnect();
}

std::optional<std::string> WebSession::recv_text() {
  if (ws_fd_ < 0) return std::nullopt;
  for (;;) {
    uint8_t hdr[2];
    if (!read_n(ws_fd_, hdr, 2)) {
      disconnect();
      return std::nullopt;
    }
    int opcode = hdr[0] & 0x0f;
    bool masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7f;
    if (len == 126) {
      uint8_t ext[2];
      if (!read_n(ws_fd_, ext, 2)) {
        disconnect();
        return std::nullopt;
      }
      len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
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
      case 0x1:  // text
        return payload;
      case 0x8:  // close
        disconnect();
        return std::nullopt;
      case 0x9: {  // ping -> pong
        std::string frame;
        frame.push_back(static_cast<char>(0x8a));
        frame.push_back(static_cast<char>(payload.size() & 0x7f));
        frame += payload;
        if (!write_all(ws_fd_, frame.data(), frame.size())) {
          disconnect();
          return std::nullopt;
        }
        break;
      }
      default:
        break;  // pong / continuation / binary: ignore
    }
  }
}

void WebSession::linger_after_final_message() {
  // Give the kernel a moment to flush the final frame before we close.
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
}

}  // namespace scribblez
