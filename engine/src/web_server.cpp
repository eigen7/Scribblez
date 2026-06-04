#include "scribblez/web_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "scribblez/tile.h"

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
      if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999u; }
      else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1u; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
      else { f = b ^ c ^ d; k = 0xCA62C1D6u; }
      uint32_t tmp = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = tmp;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  for (int i = 0; i < 5; ++i) {
    out[i * 4] = (h[i] >> 24) & 0xff;
    out[i * 4 + 1] = (h[i] >> 16) & 0xff;
    out[i * 4 + 2] = (h[i] >> 8) & 0xff;
    out[i * 4 + 3] = h[i] & 0xff;
  }
}

std::string base64(const uint8_t* data, size_t len) {
  static const char* tbl =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
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

std::string to_lower(std::string s) {
  for (char& c : s) if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
  return s;
}

// Extract a header value (case-insensitive name) from a raw HTTP request.
std::string header_value(const std::string& req, const std::string& name) {
  std::string lower = to_lower(req);
  std::string key = to_lower(name) + ":";
  size_t pos = lower.find(key);
  if (pos == std::string::npos) return "";
  pos += key.size();
  size_t end = req.find("\r\n", pos);
  std::string val = req.substr(pos, end - pos);
  size_t a = val.find_first_not_of(" \t");
  size_t b = val.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : val.substr(a, b - a + 1);
}

std::string request_path(const std::string& req) {
  size_t s = req.find(' ');
  if (s == std::string::npos) return "/";
  size_t e = req.find(' ', s + 1);
  if (e == std::string::npos) return "/";
  return req.substr(s + 1, e - s - 1);
}

std::string content_type(const std::string& path) {
  auto ends = [&](const char* ext) {
    size_t n = std::strlen(ext);
    return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
  };
  if (ends(".html")) return "text/html; charset=utf-8";
  if (ends(".js")) return "text/javascript; charset=utf-8";
  if (ends(".css")) return "text/css; charset=utf-8";
  if (ends(".json")) return "application/json";
  if (ends(".svg")) return "image/svg+xml";
  if (ends(".ico")) return "image/x-icon";
  return "application/octet-stream";
}

// --------------------------- JSON serialization --------------------------

std::string json_quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += "\"";
  return out;
}

const char* premium_code(Premium p) {
  switch (p) {
    case Premium::DLS: return "DL";
    case Premium::TLS: return "TL";
    case Premium::DWS: return "DW";
    case Premium::TWS: return "TW";
    default: return nullptr;
  }
}

// Find a tiny field in a client JSON message. Messages are small and generated
// by our own front-end, so a lenient scan suffices.
std::string json_string_field(const std::string& s, const std::string& key) {
  size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return "";
  size_t colon = s.find(':', k);
  if (colon == std::string::npos) return "";
  size_t q1 = s.find('"', colon);
  if (q1 == std::string::npos) return "";
  size_t q2 = s.find('"', q1 + 1);
  if (q2 == std::string::npos) return "";
  return s.substr(q1 + 1, q2 - q1 - 1);
}

bool json_int_field(const std::string& s, const std::string& key, long& out) {
  size_t k = s.find("\"" + key + "\"");
  if (k == std::string::npos) return false;
  size_t colon = s.find(':', k);
  if (colon == std::string::npos) return false;
  size_t i = colon + 1;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
  if (i == start) return false;
  out = std::stol(s.substr(start, i - start));
  return true;
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
  for (size_t i = 0; i < move.main_word.size(); ++i) {
    int r = move.horizontal ? sr : sr + static_cast<int>(i);
    int c = move.horizontal ? sc + static_cast<int>(i) : sc;
    bool is_blank = false;
    Square sq = board.at(r, c);
    if (!is_empty(sq)) {
      is_blank = sq.is_blank;
    } else {
      for (const auto& t : move.tiles) {
        if (t.row == r && t.col == c) { is_blank = t.is_blank; break; }
      }
    }
    char ch = move.main_word[i];
    if (is_blank && ch >= 'A' && ch <= 'Z') ch = ch - 'A' + 'a';
    word.push_back(ch);
  }

  return pos + " " + word + " " + std::to_string(move.score);
}

// --------------------------- state serialization -------------------------

std::string game_state_json(const StateView& v) {
  std::ostringstream o;
  const bool game_over = v.game_over;
  o << "{";
  o << "\"type\":" << (game_over ? "\"game_over\"" : "\"state\"");

  // board: 15x15, uppercase letters, lowercase for blanks, null for empty.
  o << ",\"board\":[";
  for (int r = 0; r < BOARD_SIZE; ++r) {
    o << (r ? ",[" : "[");
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Square sq = v.board.at(r, c);
      if (c) o << ",";
      if (is_empty(sq)) {
        o << "null";
      } else {
        char ch = letter_to_char(sq.letter);
        if (sq.is_blank) ch = ch - 'A' + 'a';
        o << "\"" << ch << "\"";
      }
    }
    o << "]";
  }
  o << "]";

  // bonuses: DL/TL/DW/TW or null.
  o << ",\"bonuses\":[";
  for (int r = 0; r < BOARD_SIZE; ++r) {
    o << (r ? ",[" : "[");
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (c) o << ",";
      const char* code = premium_code(v.board.premium_at(r, c));
      if (code) o << "\"" << code << "\""; else o << "null";
    }
    o << "]";
  }
  o << "]";

  // rack: letters then blanks ('?'), with per-tile score.
  o << ",\"rack\":[";
  bool first = true;
  for (Letter L = 0; L < 26; ++L) {
    for (int i = 0; i < v.my_rack.count(L); ++i) {
      if (!first) o << ",";
      first = false;
      o << "{\"letter\":\"" << letter_to_char(L) << "\",\"score\":" << LETTER_VALUES[L] << "}";
    }
  }
  for (int i = 0; i < v.my_rack.blanks(); ++i) {
    if (!first) o << ",";
    first = false;
    o << "{\"letter\":\"?\",\"score\":0}";
  }
  o << "]";

  o << ",\"scores\":[" << v.my_score << "," << v.opp_score << "]";
  o << ",\"player_names\":[" << json_quote(v.my_name) << "," << json_quote(v.opp_name) << "]";
  o << ",\"bag_count\":" << v.bag_size;
  o << ",\"opponent_rack_count\":" << v.opp_rack_size;
  o << ",\"your_turn\":" << (v.your_turn ? "true" : "false");
  o << ",\"game_over\":" << (game_over ? "true" : "false");

  // tile_scores map.
  o << ",\"tile_scores\":{";
  for (Letter L = 0; L < 26; ++L) {
    if (L) o << ",";
    o << "\"" << letter_to_char(L) << "\":" << LETTER_VALUES[L];
  }
  o << "}";

  // moves: only on the human's live turn.
  if (v.legal_plays && v.your_turn && !game_over) {
    o << ",\"moves\":[";
    for (size_t i = 0; i < v.legal_plays->size(); ++i) {
      const Move& m = (*v.legal_plays)[i];
      if (i) o << ",";
      o << "{\"index\":" << i
        << ",\"text\":" << json_quote(move_to_notation(v.board, m))
        << ",\"score\":" << m.score << "}";
    }
    o << "]";
  }

  if (game_over) {
    int winner = v.my_score > v.opp_score ? 0 : (v.opp_score > v.my_score ? 1 : -1);
    o << ",\"winner\":" << winner;
    o << ",\"final_scores\":[" << v.my_score << "," << v.opp_score << "]";
  }

  o << "}";
  return o.str();
}

// ------------------------------ WebSession -------------------------------

WebSession::WebSession(int port, std::string web_dir)
    : port_(port), web_dir_(std::move(web_dir)) {
  std::signal(SIGPIPE, SIG_IGN);
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
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

void WebSession::serve_static(int fd, const std::string& path) {
  // Map and sanitize the path.
  std::string rel = path == "/" ? "/index.html" : path;
  if (size_t q = rel.find('?'); q != std::string::npos) rel = rel.substr(0, q);
  if (rel.find("..") != std::string::npos) {
    std::string body = "Forbidden";
    std::string resp = "HTTP/1.1 403 Forbidden\r\nConnection: close\r\n"
                       "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    write_all(fd, resp.data(), resp.size());
    return;
  }

  std::ifstream f(web_dir_ + rel, std::ios::binary);
  if (!f) {
    std::string body = "Not found: " + rel +
                       "\n(Did you build the web UI? See web/README or run "
                       "`npm --prefix web run build`.)";
    std::string resp = "HTTP/1.1 404 Not Found\r\nConnection: close\r\n"
                       "Content-Type: text/plain\r\nContent-Length: " +
                       std::to_string(body.size()) + "\r\n\r\n" + body;
    write_all(fd, resp.data(), resp.size());
    return;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string body = ss.str();
  std::string resp = "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: " +
                     content_type(rel) + "\r\nContent-Length: " +
                     std::to_string(body.size()) + "\r\n\r\n";
  write_all(fd, resp.data(), resp.size());
  write_all(fd, body.data(), body.size());
}

bool WebSession::do_handshake(int fd, const std::string& request) {
  std::string key = header_value(request, "Sec-WebSocket-Key");
  if (key.empty()) return false;
  uint8_t digest[20];
  sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11", digest);
  std::string accept = base64(digest, 20);
  std::string resp = "HTTP/1.1 101 Switching Protocols\r\n"
                     "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                     "Sec-WebSocket-Accept: " + accept + "\r\n\r\n";
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
    if (req.empty()) { ::close(conn); continue; }

    if (to_lower(header_value(req, "Upgrade")) == "websocket") {
      if (do_handshake(conn, req)) {
        ws_fd_ = conn;
        return true;
      }
      ::close(conn);
      continue;
    }
    serve_static(conn, request_path(req));
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
    if (!read_n(ws_fd_, hdr, 2)) { disconnect(); return std::nullopt; }
    int opcode = hdr[0] & 0x0f;
    bool masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7f;
    if (len == 126) {
      uint8_t ext[2];
      if (!read_n(ws_fd_, ext, 2)) { disconnect(); return std::nullopt; }
      len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
    } else if (len == 127) {
      uint8_t ext[8];
      if (!read_n(ws_fd_, ext, 8)) { disconnect(); return std::nullopt; }
      len = 0;
      for (int i = 0; i < 8; ++i) len = (len << 8) | ext[i];
    }
    uint8_t mask[4] = {0, 0, 0, 0};
    if (masked && !read_n(ws_fd_, mask, 4)) { disconnect(); return std::nullopt; }
    std::string payload(len, '\0');
    if (len && !read_n(ws_fd_, payload.data(), len)) { disconnect(); return std::nullopt; }
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
        if (!write_all(ws_fd_, frame.data(), frame.size())) { disconnect(); return std::nullopt; }
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

// ------------------------------ HumanWebAgent ----------------------------

HumanWebAgent::HumanWebAgent(WebSession& session, std::string my_name,
                             std::string opp_name)
    : session_(session), my_name_(std::move(my_name)), opp_name_(std::move(opp_name)) {}

Move HumanWebAgent::choose(const AgentContext& ctx, std::mt19937_64&) {
  StateView view{ctx.board,
                 ctx.my_rack,
                 ctx.my_score,
                 ctx.opp_score,
                 ctx.bag_size,
                 ctx.opp_rack_size,
                 my_name_,
                 opp_name_,
                 &ctx.legal_plays,
                 /*your_turn=*/true,
                 /*game_over=*/false};
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_.connected() && !session_.wait_for_client()) {
      Move m;
      m.type = MoveType::PASS;
      return m;
    }
    session_.send_text(msg);
    for (;;) {
      auto in = session_.recv_text();
      if (!in) break;  // disconnected: re-send on reconnect
      const std::string type = json_string_field(*in, "type");
      if (type == "move") {
        long idx = -1;
        if (json_int_field(*in, "index", idx) && idx >= 0 &&
            static_cast<size_t>(idx) < ctx.legal_plays.size()) {
          return ctx.legal_plays[static_cast<size_t>(idx)];
        }
      } else if (type == "pass") {
        Move m;
        m.type = MoveType::PASS;
        return m;
      } else if (type == "exchange") {
        // Optional: front-end may send {"type":"exchange","letters":"AB?"}.
        Move m;
        m.type = MoveType::EXCHANGE;
        const std::string letters = json_string_field(*in, "letters");
        for (char c : letters) {
          Letter L = (c == '?' || (c >= 'a' && c <= 'z')) ? BLANK : char_to_letter(c);
          if (ctx.my_rack.count(L) > 0) m.exchanged.push_back(L);
        }
        if (!m.exchanged.empty()) return m;
      }
      // Unknown / invalid: keep waiting for a usable message.
    }
  }
}

}  // namespace scribblez
