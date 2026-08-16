#include "game/board.h"
#include "game/glyph.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "serve/web_server.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {
namespace {

char upper_ch(char c) {
  if (c >= 'a' && c <= 'z') return char(c - 'a' + 'A');
  return c;
}

int int_field(const boost::json::object& o, const char* key, int fallback = -1) {
  auto it = o.find(key);
  if (it == o.end() || !it->value().is_int64()) return fallback;
  return it->value().as_int64();
}

std::string str_field(const boost::json::object& o, const char* key) {
  auto it = o.find(key);
  if (it == o.end() || !it->value().is_string()) return "";
  return std::string(it->value().as_string().c_str());
}

bool bool_field(const boost::json::object& o, const char* key, bool fallback = false) {
  auto it = o.find(key);
  if (it == o.end() || !it->value().is_bool()) return fallback;
  return it->value().as_bool();
}

// The letter Tile named by a one-character string, or EMPTY_SQUARE if it is not
// a letter A..Z. Designated blanks carry their shown letter here; the blank-ness
// is tracked separately.
Tile letter_tile(const std::string& s) {
  if (s.empty()) return EMPTY_SQUARE;
  const char c = upper_ch(s[0]);
  if (c < 'A' || c > 'Z') return EMPTY_SQUARE;
  return Tile::from_char(c);
}

TileCounts full_bag() {
  TileCounts bag;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    for (int i = 0; i < TILE_COUNTS[L]; ++i) bag.add(L);
  }
  for (int i = 0; i < TILE_COUNTS[BLANK]; ++i) bag.add(BLANK);
  return bag;
}

boost::json::array board_grid(const Board& board) {
  boost::json::array grid;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    boost::json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (g.is_empty()) {
        row.emplace_back(nullptr);
      } else {
        char ch = g.letter().to_char();
        if (g.is_blank()) ch = char(ch - 'A' + 'a');
        row.emplace_back(std::string(1, ch));
      }
    }
    grid.emplace_back(std::move(row));
  }
  return grid;
}

boost::json::array bonus_grid(const Board& board) {
  boost::json::array grid;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    boost::json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const char* code = board.premium_at(r, c).code();
      row.emplace_back(code ? boost::json::value(code) : boost::json::value(nullptr));
    }
    grid.emplace_back(std::move(row));
  }
  return grid;
}

boost::json::object tile_score_map() {
  boost::json::object scores;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    scores[std::string(1, L.to_char())] = TILE_VALUES[L];
  }
  return scores;
}

// Remaining bag contents as a JSON array of {letter, score, count}: each present
// letter (and the blank as "?") with a positive count.
boost::json::array bag_tiles_json(const TileCounts& bag) {
  boost::json::array bag_tiles;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    const int count = bag.count(L);
    if (count <= 0) continue;
    bag_tiles.emplace_back(boost::json::object{
      {"letter", std::string(1, L.to_char())}, {"score", L.value()}, {"count", count}});
  }
  const int blanks = bag.count(BLANK);
  if (blanks > 0) {
    bag_tiles.emplace_back(boost::json::object{{"letter", "?"}, {"score", 0}, {"count", blanks}});
  }
  return bag_tiles;
}

// A board word that the dictionary rejects, with its starting coordinate in GCG
// notation (row number first across, column letter first down).
struct InvalidWord {
  std::string word;
  std::string square;
  std::string direction;  // "across" or "down"
};

// A freeform 15x15 board the user fills however they like, drawing tiles from a
// shared bag, plus a word-legality check over every contiguous run.
class BoardEditor {
 public:
  explicit BoardEditor(const Dictionary& dict) : dict_(dict) { reset(); }

  void reset() {
    board_ = Board();
    bag_ = full_bag();
    validation_.reset();
    status_.clear();
  }

  void place(int row, int col, const std::string& letter_str, bool is_blank) {
    validation_.reset();
    if (!board_.in_bounds(row, col)) {
      status_ = "Invalid square";
      return;
    }
    if (!board_.at(row, col).is_empty()) {
      status_ = "That square already holds a tile";
      return;
    }
    const Tile letter = letter_tile(letter_str);
    if (letter.is_empty()) {
      status_ = "Invalid letter";
      return;
    }
    const Tile bag_tile = is_blank ? BLANK : letter;
    if (bag_.count(bag_tile) <= 0) {
      status_ = std::string("No ") + (is_blank ? "blank" : std::string(1, letter.to_char())) +
                " tiles left in the bag";
      return;
    }
    bag_.remove(bag_tile);
    board_.set(row, col, Glyph::played(letter, is_blank));
    status_.clear();
  }

  void remove(int row, int col) {
    validation_.reset();
    if (!board_.in_bounds(row, col)) return;
    const Glyph g = board_.at(row, col);
    if (g.is_empty()) return;
    bag_.add(g.is_blank() ? BLANK : g.letter());
    board_.set(row, col, Glyph::empty());
    status_.clear();
  }

  void validate() {
    std::vector<InvalidWord> invalid;
    for (int line = 0; line < BOARD_SIZE; ++line) {
      scan_line(/*vertical=*/false, line, &invalid);
      scan_line(/*vertical=*/true, line, &invalid);
    }
    validation_ = ValidationResult{invalid.empty(), std::move(invalid)};
    status_.clear();
  }

  boost::json::object state_json() const {
    boost::json::object o;
    o["type"] = "board_state";
    o["board"] = board_grid(board_);
    o["bonuses"] = bonus_grid(board_);
    o["bag_tiles"] = bag_tiles_json(bag_);
    o["bag_count"] = bag_.size();
    o["tile_scores"] = tile_score_map();
    o["lexicon"] = Lexicon::instance().name();
    o["status"] = status_;
    o["validation"] = validation_json();
    return o;
  }

 private:
  struct ValidationResult {
    bool valid = false;
    std::vector<InvalidWord> invalid;
  };

  Glyph oriented_at(bool vertical, int line, int index) const {
    return vertical ? board_.at(index, line) : board_.at(line, index);
  }

  // Scan one row (or column when `vertical`) for maximal contiguous runs and
  // record every run of length >= 2 that the dictionary rejects.
  void scan_line(bool vertical, int line, std::vector<InvalidWord>* out) const {
    int i = 0;
    while (i < BOARD_SIZE) {
      if (oriented_at(vertical, line, i).is_empty()) {
        ++i;
        continue;
      }
      const int start = i;
      std::string word;
      while (i < BOARD_SIZE && !oriented_at(vertical, line, i).is_empty()) {
        word.push_back(oriented_at(vertical, line, i).letter().to_char());
        ++i;
      }
      if (int(word.size()) >= 2 && !dict_.contains(word)) {
        out->push_back(make_invalid(vertical, line, start, word));
      }
    }
  }

  static InvalidWord make_invalid(bool vertical, int line, int start, const std::string& word) {
    const int row = vertical ? start : line;
    const int col = vertical ? line : start;
    const std::string col_letter(1, char('A' + col));
    const std::string row_num = std::to_string(row + 1);
    return InvalidWord{word, vertical ? col_letter + row_num : row_num + col_letter,
                       vertical ? "down" : "across"};
  }

  boost::json::value validation_json() const {
    if (!validation_.has_value()) return boost::json::value(nullptr);
    boost::json::array words;
    for (const InvalidWord& w : validation_->invalid) {
      words.emplace_back(
        boost::json::object{{"word", w.word}, {"square", w.square}, {"direction", w.direction}});
    }
    return boost::json::object{{"valid", validation_->valid}, {"invalid_words", std::move(words)}};
  }

  const Dictionary& dict_;
  Board board_;
  TileCounts bag_;
  std::optional<ValidationResult> validation_;
  std::string status_;
};

void handle_message(BoardEditor& editor, const boost::json::object& obj) {
  const std::string type = str_field(obj, "type");
  if (type == "place") {
    editor.place(int_field(obj, "row"), int_field(obj, "col"), str_field(obj, "letter"),
                 bool_field(obj, "isBlank"));
  } else if (type == "remove") {
    editor.remove(int_field(obj, "row"), int_field(obj, "col"));
  } else if (type == "clear") {
    editor.reset();
  } else if (type == "validate") {
    editor.validate();
  }
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    int ws_port = 8083;
    int vite_port = 5175;
    std::string web_dir = "web";

    po::options_description desc("board_tool options");
    desc.add_options()("help,h", "show this help message and exit")(
      "port", po::value<int>(&ws_port)->default_value(ws_port), "engine WebSocket port")(
      "vite-port", po::value<int>(&vite_port)->default_value(vite_port), "browser UI port")(
      "web-dir", po::value<std::string>(&web_dir)->default_value(web_dir),
      "front-end package dir (cwd of npm run dev)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::load_dictionary_or_throw();
    scribblez::WebSession session(ws_port);
    scribblez::ViteDevServer vite(web_dir, vite_port, ws_port, "board", "board", 5175);
    if (!vite.wait_until_ready()) {
      throw scribblez::util::Exception("the Vite dev server did not start; see web/.vite-dev.log");
    }

    std::cerr << "\nBoard tool ready at " << vite.url()
              << " (lexicon: " << scribblez::Lexicon::instance().name() << ")\n";
    std::string cmd = "xdg-open '" + vite.url() + "' >/dev/null 2>&1 &";
    int rc = std::system(cmd.c_str());
    (void)rc;

    scribblez::BoardEditor editor(dict);

    while (true) {
      if (!session.connected()) {
        if (!session.wait_for_client()) break;
      }
      session.send_text(boost::json::serialize(editor.state_json()));
      for (;;) {
        auto in = session.recv_text();
        if (!in.has_value()) break;
        boost::json::value parsed;
        try {
          parsed = boost::json::parse(*in);
        } catch (const std::exception&) {
          continue;
        }
        if (!parsed.is_object()) continue;
        scribblez::handle_message(editor, parsed.as_object());
        session.send_text(boost::json::serialize(editor.state_json()));
      }
    }

    return 0;
  } catch (...) {
    return scribblez::util::main_exit_code();
  }
}
