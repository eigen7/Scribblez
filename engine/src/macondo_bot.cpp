#include "scribblez/macondo_bot.h"

#include "scribblez/board.h"
#include "scribblez/glyph.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <boost/process.hpp>

#include <array>
#include <cctype>
#include <string>
#include <system_error>

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

// Full CGP for the on-move player (their rack/score first), forcing NWL23 to
// match Scribblez (this Macondo build defaults to NWL20).
std::string to_cgp(const Board& board, const Rack& rack, int my_score, int opp_score) {
  return board_fen(board) + " " + rack.to_string() + "/ " + std::to_string(my_score) + "/" +
         std::to_string(opp_score) + " 0 lex NWL23;";
}

// --------------------------- the Macondo process -------------------------

// One persistent `macondo` shell, shared by all HastyBot seats. Loaded once
// (lexicon/leaves), then driven a turn at a time over its stdin/stdout.
class Macondo {
 public:
  static Macondo& get(const std::string& binary) {
    static Macondo instance(binary);
    return instance;
  }

  // The rank-1 (best static equity) play for a position, as Macondo prints it
  // (e.g. "8F UNITERS", "(exch S)", "(Pass)"). Empty on failure.
  std::string best_play(const std::string& cgp) {
    to_ << "load cgp " << cgp << "\ngen\nSCRIBBLEZ_DONE\n" << std::flush;
    std::string line, play;
    while (std::getline(from_, line)) {
      strip_ansi(line);
      if (line.find("SCRIBBLEZ_DONE") != std::string::npos) break;  // end-of-response sentinel
      size_t s = line.find_first_not_of(" \t");
      if (s != std::string::npos && line.compare(s, 2, "1:") == 0) {
        play = move_column(line.substr(s + 2));
      }
    }
    return play;
  }

  ~Macondo() {
    std::error_code ec;
    child_.terminate(ec);
    child_.wait(ec);
  }

 private:
  explicit Macondo(const std::string& binary)
      : child_(binary, bp::std_in<to_, bp::std_out> from_, bp::std_err > bp::null) {}

  static void strip_ansi(std::string& s) {
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

  // The Move column of a gen row: from the first non-space to the first 2-space
  // gap (which separates it from the Leave/Score/Equity columns).
  static std::string move_column(const std::string& rest) {
    size_t s = rest.find_first_not_of(" \t");
    if (s == std::string::npos) return "";
    size_t e = rest.find("  ", s);
    return rest.substr(s, (e == std::string::npos ? rest.size() : e) - s);
  }

  bp::opstream to_;
  bp::ipstream from_;
  bp::child child_;
};

// --------------------------- play -> Move --------------------------------

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

Move parse_play(const std::string& play, const AgentContext& ctx) {
  Move pass;
  pass.type = MoveType::PASS;
  if (play.empty() || play.find("Pass") != std::string::npos) return pass;

  if (play.rfind("(exch", 0) == 0) {
    Move m;
    m.type = MoveType::EXCHANGE;
    size_t sp = play.find(' '), rp = play.find(')');
    int gi = 0;
    for (size_t i = (sp == std::string::npos ? play.size() : sp + 1);
         i < rp && i < play.size() && gi < RACK_SIZE; ++i) {
      Tile t = play[i] == '?' ? BLANK : Tile::from_char(play[i]);
      m.glyphs[gi++] = Glyph::exchanging(t);
    }
    return gi > 0 ? m : pass;
  }

  // Tile placement: "<coord> <word>".
  size_t sp = play.find(' ');
  if (sp == std::string::npos) return pass;
  bool horizontal;
  int sr, sc;
  if (!parse_coord(play.substr(0, sp), horizontal, sr, sc)) return pass;

  Move m;
  m.type = MoveType::PLAY;
  m.horizontal = horizontal;
  m.start_row = static_cast<int8_t>(sr);
  m.start_col = static_cast<int8_t>(sc);
  int gi = 0;
  for (char ch : play.substr(sp + 1)) {
    if (ch == '.') continue;  // played-through existing tile
    bool blank = ch >= 'a' && ch <= 'z';
    if (gi < RACK_SIZE) m.glyphs[gi++] = Glyph::played(Tile::from_char(ch), blank);
  }

  // Return the matching legal play (gives the correct Scribblez score and
  // confirms Macondo's choice is legal in our engine).
  for (const Move& lp : ctx.legal_plays) {
    if (lp.start_row == m.start_row && lp.start_col == m.start_col &&
        lp.horizontal == m.horizontal && lp.glyphs == m.glyphs) {
      return lp;
    }
  }
  return pass;  // not a legal Scribblez play here -> pass rather than cheat
}

}  // namespace

HastyBotAgent::HastyBotAgent(std::string macondo_binary, std::string name)
    : macondo_binary_(std::move(macondo_binary)), name_(std::move(name)) {}

Move HastyBotAgent::choose(const AgentContext& ctx, std::mt19937_64&) {
  std::string cgp = to_cgp(ctx.board, ctx.my_rack, ctx.my_score, ctx.opp_score);
  std::string play = Macondo::get(macondo_binary_).best_play(cgp);
  return parse_play(play, ctx);
}

}  // namespace scribblez
