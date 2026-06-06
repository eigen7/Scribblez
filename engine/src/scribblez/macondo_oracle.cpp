#include "scribblez/macondo_oracle.h"

#include "scribblez/glyph.h"
#include "scribblez/tile.h"

#include <boost/process.hpp>

#include <cctype>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace scribblez {

namespace {

namespace bp = boost::process;

// --------------------------- CGP encoding --------------------------------

// Board as a CGP fen: rows joined by '/', empty runs as digits, tiles as
// letters (lowercase for a designated blank already on the board).
std::string board_fen(const Board& b) {
  std::string fen;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    int empty = 0;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = b.at(r, c);
      if (g.is_empty()) {
        ++empty;
        continue;
      }
      if (empty) {
        fen += std::to_string(empty);
        empty = 0;
      }
      char ch = g.letter().to_char();
      fen.push_back(g.is_blank() ? static_cast<char>(ch - 'A' + 'a') : ch);
    }
    if (empty) fen += std::to_string(empty);
    if (r + 1 < BOARD_SIZE) fen.push_back('/');
  }
  return fen;
}

// Full CGP for the on-move player (their rack/score first), explicitly setting
// the `lex` clause to whatever Scribblez is using (so it matches the kwg the
// engine loaded, regardless of macondo's own DEFAULT_LEXICON).
std::string to_cgp(const Board& board, const Rack& rack, int my_score, int opp_score,
                   const std::string& lexicon) {
  return board_fen(board) + " " + rack.to_string() + "/ " + std::to_string(my_score) + "/" +
         std::to_string(opp_score) + " 0 lex " + lexicon + ";";
}

// ----------------------- Macondo line parsing ----------------------------

void strip_ansi(std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '\033') {  // skip a CSI escape "\033[ ... <letter>"
      i += 2;
      while (i < s.size() && !std::isalpha(static_cast<unsigned char>(s[i]))) ++i;
      if (i < s.size()) ++i;
    } else {
      out.push_back(s[i++]);
    }
  }
  s.swap(out);
}

// Split `rest` (the line content after the `N:` prefix) into (move_text,
// equity). The move column ends at the first 2-space gap; the equity is the
// last whitespace-separated token of the remainder. Returns false if the line
// doesn't look like a gen-table row.
bool parse_gen_row(const std::string& rest, std::string& move, double& equity) {
  size_t s = rest.find_first_not_of(" \t");
  if (s == std::string::npos) return false;
  size_t e = rest.find("  ", s);
  if (e == std::string::npos) return false;  // need a 2-space gap separating leave/score/equity
  move = rest.substr(s, e - s);

  // Last whitespace-separated token of the remainder is equity.
  std::string tail = rest.substr(e);
  size_t end = tail.find_last_not_of(" \t");
  if (end == std::string::npos) return false;
  size_t start = tail.find_last_of(" \t", end);
  std::string tok = tail.substr(start == std::string::npos ? 0 : start + 1, end + 1 - (start == std::string::npos ? 0 : start + 1));
  try {
    equity = std::stod(tok);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

// ----------------------- play-text -> Move match -------------------------

// Parse a coordinate: "8F" (number first) is horizontal/across; "F8" (letter
// first) is vertical/down.
bool parse_coord(const std::string& s, bool& horizontal, int& sr, int& sc) {
  if (s.empty()) return false;
  auto read_num = [&](size_t& i) {
    int n = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
      n = n * 10 + (s[i++] - '0');
    return n;
  };
  size_t i = 0;
  if (std::isdigit(static_cast<unsigned char>(s[0]))) {
    int row = read_num(i);
    if (i >= s.size()) return false;
    horizontal = true;
    sr = row - 1;
    sc = s[i] - 'A';
  } else {
    sc = s[0] - 'A';
    i = 1;
    horizontal = false;
    sr = read_num(i) - 1;
  }
  return sr >= 0 && sr < BOARD_SIZE && sc >= 0 && sc < BOARD_SIZE;
}

// Build a fingerprint (start_row, start_col, horizontal, glyphs) for a Move,
// suitable for an unordered_map lookup. Returns nullopt if `play` is not a
// tile-placement we can match (passes/exchanges).
struct MoveKey {
  int8_t start_row;
  int8_t start_col;
  bool horizontal;
  std::array<Glyph, RACK_SIZE> glyphs;
  bool operator==(const MoveKey& o) const noexcept {
    return start_row == o.start_row && start_col == o.start_col && horizontal == o.horizontal &&
           glyphs == o.glyphs;
  }
};
struct MoveKeyHash {
  size_t operator()(const MoveKey& k) const noexcept {
    // Mix the small fields and each glyph byte into a 64-bit hash. The glyph
    // representation is small (~7 bytes), so a per-byte fold is plenty.
    size_t h = static_cast<size_t>(k.start_row) * 31 + static_cast<size_t>(k.start_col);
    h = h * 31 + (k.horizontal ? 1u : 0u);
    const auto* p = reinterpret_cast<const unsigned char*>(k.glyphs.data());
    for (size_t i = 0; i < sizeof(k.glyphs); ++i) h = h * 131u + p[i];
    return h;
  }
};

std::optional<MoveKey> parse_play_key(const std::string& play) {
  if (play.empty() || play.find("Pass") != std::string::npos) return std::nullopt;
  if (play.rfind("(exch", 0) == 0) return std::nullopt;  // exchanges aren't in legal_plays

  // Tile placement: "<coord> <word>".
  size_t sp = play.find(' ');
  if (sp == std::string::npos) return std::nullopt;
  bool horizontal;
  int sr, sc;
  if (!parse_coord(play.substr(0, sp), horizontal, sr, sc)) return std::nullopt;

  MoveKey k{};
  k.start_row = static_cast<int8_t>(sr);
  k.start_col = static_cast<int8_t>(sc);
  k.horizontal = horizontal;
  int gi = 0;
  for (char ch : play.substr(sp + 1)) {
    if (ch == '.') continue;  // played-through existing tile
    bool blank = ch >= 'a' && ch <= 'z';
    if (gi < RACK_SIZE) k.glyphs[gi++] = Glyph::played(Tile::from_char(ch), blank);
  }
  return k;
}

MoveKey key_of(const Move& m) {
  MoveKey k{};
  k.start_row = m.start_row;
  k.start_col = m.start_col;
  k.horizontal = m.horizontal;
  k.glyphs = m.glyphs;
  return k;
}

// Generate up to this many plays per evaluation. Large enough to cover any
// realistic Scrabble position (legal-play counts top out near ~1500); the
// extra slack is cheap because Macondo's printer is the bottleneck, not its
// move generator.
constexpr int kGenLimit = 10000;

}  // namespace

// --------------------------- the subprocess ------------------------------

struct MacondoOracle::Impl {
  bp::opstream to;
  bp::ipstream from;
  bp::child child;

  explicit Impl(const std::string& binary)
      : child(binary, bp::std_in<to, bp::std_out> from, bp::std_err > bp::null) {}

  ~Impl() {
    std::error_code ec;
    child.terminate(ec);
    child.wait(ec);
  }
};

// --------------------------- ctor / params -------------------------------

MacondoOracle::Params::Params() : binary_path("/workspace/mount/macondo/bin/shell") {}

MacondoOracle::MacondoOracle(const Params& params)
    : binary_(params.binary_path), lexicon_(params.lexicon) {}

MacondoOracle::~MacondoOracle() = default;

void MacondoOracle::ensure_started() {
  if (impl_) return;
  impl_ = std::make_unique<Impl>(binary_);
}

// --------------------------- evaluate ------------------------------------

MacondoOracle::EvalResult MacondoOracle::evaluate(const Board& board, const Rack& my_rack,
                                                  int my_score, int opp_score,
                                                  const std::vector<Move>& legal_plays) {
  ensure_started();

  EvalResult result;
  result.equities.assign(legal_plays.size(), std::nullopt);

  // Index legal_plays by their move-identity key so we can map each Macondo
  // gen-row back to a legal index in O(1).
  std::unordered_map<MoveKey, int, MoveKeyHash> by_key;
  by_key.reserve(legal_plays.size() * 2);
  for (size_t i = 0; i < legal_plays.size(); ++i) {
    const Move& m = legal_plays[i];
    if (m.type != MoveType::PLAY) continue;
    by_key.emplace(key_of(m), static_cast<int>(i));
  }

  // Drive Macondo: load the position, generate N plays, print a sentinel so
  // we know when its response ends.
  std::string cgp = to_cgp(board, my_rack, my_score, opp_score, lexicon_);
  impl_->to << "load cgp " << cgp << "\ngen " << kGenLimit << "\nSCRIBBLEZ_DONE\n" << std::flush;

  std::string line;
  while (std::getline(impl_->from, line)) {
    strip_ansi(line);
    if (line.find("SCRIBBLEZ_DONE") != std::string::npos) break;

    // Match "  RANK: ..." lines (the gen table).
    size_t s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    size_t colon = line.find(':', s);
    if (colon == std::string::npos) continue;
    bool all_digits = colon > s;
    for (size_t i = s; i < colon && all_digits; ++i) {
      if (!std::isdigit(static_cast<unsigned char>(line[i]))) all_digits = false;
    }
    if (!all_digits) continue;

    int rank = std::atoi(line.substr(s, colon - s).c_str());
    std::string move_text;
    double equity = 0;
    if (!parse_gen_row(line.substr(colon + 1), move_text, equity)) continue;

    auto key = parse_play_key(move_text);
    if (!key) continue;  // pass / exchange / unparseable
    auto it = by_key.find(*key);
    if (it == by_key.end()) continue;

    int idx = it->second;
    result.equities[idx] = equity;
    if (rank == 1) result.best_index = idx;
  }

  // Opening-symmetry fill: on an empty board, every 1D play "8D STEARIN" has
  // a vertical mirror "H4 STEARIN" that scores identically (no cross-words
  // either way) and forms an isomorphic position; Macondo dedups these and
  // returns only one orientation, leaving the other unmatched in our
  // legal_plays. Propagate the equity from a play's transposed sibling --
  // gated on equal scores, so genuinely-distinct mid-game H/V plays (which
  // diverge via cross-word scoring) never bleed equity into each other.
  for (size_t i = 0; i < legal_plays.size(); ++i) {
    if (result.equities[i].has_value()) continue;
    const Move& m = legal_plays[i];
    if (m.type != MoveType::PLAY) continue;
    MoveKey tk = key_of(m);
    std::swap(tk.start_row, tk.start_col);
    tk.horizontal = !tk.horizontal;
    auto it = by_key.find(tk);
    if (it == by_key.end()) continue;
    int j = it->second;
    if (!result.equities[j].has_value()) continue;
    if (legal_plays[j].score != m.score) continue;
    result.equities[i] = result.equities[j];
  }

  return result;
}

}  // namespace scribblez
