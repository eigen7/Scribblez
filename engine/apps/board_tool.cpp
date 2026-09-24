// board_tool: a browser-based freeform board editor. Place any tiles anywhere
// (drawn from one shared bag, blanks included) and ask the lexicon which words
// on the board are invalid. Useful for composing positions by hand. Like a
// human seat in play_game, the tool launches the web UI's Vite dev server and
// opens it; the UI talks to this process over a WebSocket.
//
//   board_tool [--lexicon NWL23] [--port 8083] [--vite-port 5175]

#include "game/board.h"
#include "game/glyph.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"
#include "lexicon/lexicon.h"
#include "serve/client_message.h"
#include "serve/position_json.h"
#include "serve/web_server.h"
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

// The letter Tile named by a one-character string, or EMPTY_SQUARE if it is not
// a letter A..Z. For a blank this is the letter it stands for; whether the tile
// is a blank travels separately.
Tile letter_tile(const std::string& s) {
  if (s.empty()) return EMPTY_SQUARE;
  const char c = upper_ch(s[0]);
  if (c < 'A' || c > 'Z') return EMPTY_SQUARE;
  return Tile::from_char(c);
}

// A board word the dictionary rejects. `square` is its starting square in GCG
// notation: row number first for across words, column letter first for down.
struct InvalidWord {
  std::string word;
  std::string square;
  std::string direction;  // "across" or "down"
};

// The editor's state: the board, the bag its tiles come from, and the result
// of the last word check.
class BoardEditor {
 public:
  explicit BoardEditor(const Dictionary& dict) : dict_(dict) { reset(); }

  void reset() {
    board_ = Board();
    bag_ = TileCounts::full_distribution();
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

  // Record every run of two or more tiles in this row (or column, when
  // `vertical`) that the dictionary rejects.
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
      "front-end package directory (where npm run dev is started)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::util::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::load_dictionary_or_throw();
    scribblez::WebSession session(ws_port);
    scribblez::ViteDevServer vite(web_dir, vite_port, ws_port, "board", "board", 5175);
    vite.wait_until_ready();

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
