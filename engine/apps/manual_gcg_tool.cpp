#include "scribblez/cli.h"
#include "scribblez/exception.h"
#include "scribblez/game.h"
#include "scribblez/game_runner.h"
#include "scribblez/gcg_writer.h"
#include "scribblez/lexicon.h"
#include "scribblez/movegen.h"
#include "scribblez/position_json.h"
#include "scribblez/web_server.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

constexpr int kRackSlots = 7;
constexpr int kMaxNameLen = 18;

using RackSlots = std::array<std::optional<Tile>, kRackSlots>;

struct ManualTilePlacement {
  int row = -1;
  int col = -1;
  Tile tile = EMPTY_SQUARE;
  char letter = '\0';
  bool is_blank = false;
  bool from_rack = false;
  int rack_slot = -1;
};

struct ManualTurn {
  TurnRecord record;
  bool include_rack_before = false;
  std::string notation;
  RackSlots rack_before_slots;
  std::array<RackSlots, 2> racks_after_turn;
  std::optional<std::string> exchange_field;
};

char upper_ch(char c) {
  if (c >= 'a' && c <= 'z') return static_cast<char>(c - 'a' + 'A');
  return c;
}

std::string now_string() {
  using clock = std::chrono::system_clock;
  const std::time_t t = clock::to_time_t(clock::now());
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  std::ostringstream o;
  o << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %Z");
  return o.str();
}

std::string trim_name(const std::string& in) {
  if (in.empty()) return in;
  return in.substr(0, kMaxNameLen);
}

Tile tile_from_letter(const std::string& s, bool is_blank) {
  if (is_blank) return BLANK;
  if (s.empty()) return EMPTY_SQUARE;
  const char c = upper_ch(s[0]);
  if (c < 'A' || c > 'Z') return EMPTY_SQUARE;
  return Tile::from_char(c);
}

boost::json::array board_grid(const Board& board) {
  boost::json::array grid;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    boost::json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = board.at(r, c);
      if (g.is_empty()) {
        row.emplace_back(nullptr);
      } else {
        char ch = g.letter().to_char();
        if (g.is_blank()) ch = static_cast<char>(ch - 'A' + 'a');
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

int int_field(const boost::json::object& o, const char* key, int fallback = -1) {
  auto it = o.find(key);
  if (it == o.end() || !it->value().is_int64()) return fallback;
  return static_cast<int>(it->value().as_int64());
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

std::string gcg_rack_field(const RackSlots& slots) {
  std::string out;
  out.reserve(kRackSlots);
  for (const auto& tile : slots) {
    out.push_back(tile.has_value() ? tile->to_char() : '_');
  }
  return out;
}

bool has_known_tiles(const RackSlots& slots) {
  for (const auto& tile : slots) {
    if (tile.has_value()) return true;
  }
  return false;
}

std::optional<std::string> maybe_rack_pragma(const RackSlots& slots) {
  if (!has_known_tiles(slots)) return std::nullopt;
  return gcg_rack_field(slots);
}

int board_tile_count(const Board& board) {
  int n = 0;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!board.at(r, c).is_empty()) ++n;
    }
  }
  return n;
}

class ManualGame {
 public:
  explicit ManualGame(const Dictionary& dict) : dict_(dict), movegen_(board_, dict_) {
    names_[0] = "Player 1";
    names_[1] = "Player 2";
    reset();
  }

  boost::json::object state_json() const {
    boost::json::object o;
    o["type"] = "manual_state";
    o["board"] = board_grid(board_);
    o["bonuses"] = bonus_grid(board_);
    o["scores"] = {scores_[0], scores_[1]};
    o["player_names"] = {names_[0], names_[1]};
    o["current_player"] = turn_player_;
    const int bag_count = std::max(0, 100 - board_tile_count(board_) - 14);
    o["bag_count"] = bag_count;
    o["rack_known_counts"] = {known_count(0), known_count(1)};
    o["lexicon"] = Lexicon::instance().name();
    o["max_name_len"] = kMaxNameLen;
    o["status"] = status_;
    o["default_export_path"] = "/workspace/repo/manual_game.gcg";
    o["tile_scores"] = tile_score_map();

    boost::json::array racks;
    for (int p = 0; p < 2; ++p) {
      boost::json::array r;
      for (int s = 0; s < kRackSlots; ++s) {
        if (racks_[p][s].has_value()) {
          const Tile t = racks_[p][s].value();
          r.emplace_back(boost::json::object{
            {"letter", std::string(1, t.to_char())}, {"score", t.value()}, {"known", true}});
        } else {
          r.emplace_back(boost::json::object{{"letter", "?"}, {"score", 0}, {"known", false}});
        }
      }
      racks.emplace_back(std::move(r));
    }
    o["racks"] = std::move(racks);

    boost::json::array bag_tiles;
    for (Tile L = Tile::of(0); L < 26; ++L) {
      const int count = bag_.count(L);
      if (count <= 0) continue;
      bag_tiles.emplace_back(boost::json::object{
        {"letter", std::string(1, L.to_char())}, {"score", L.value()}, {"count", count}});
    }
    const int blanks = bag_.count(BLANK);
    if (blanks > 0) {
      bag_tiles.emplace_back(boost::json::object{{"letter", "?"}, {"score", 0}, {"count", blanks}});
    }
    o["bag_tiles"] = std::move(bag_tiles);

    boost::json::array turn_list;
    for (const ManualTurn& t : turns_) {
      turn_list.emplace_back(boost::json::object{
        {"player", t.record.player},
        {"text", t.notation},
        {"score", t.record.score_delta},
        {"cumulative", {t.record.cumulative_scores[0], t.record.cumulative_scores[1]}},
        {"rack", t.include_rack_before ? boost::json::value(t.record.rack_before.to_string())
                                       : boost::json::value(nullptr)}});
    }
    o["turns"] = std::move(turn_list);

    return o;
  }

  void reset() {
    board_ = Board();
    scores_ = {0, 0};
    turn_player_ = 0;
    turns_.clear();
    status_ = "";
    for (int p = 0; p < 2; ++p) {
      for (int s = 0; s < kRackSlots; ++s) racks_[p][s].reset();
    }
    bag_ = TileCounts();
    for (Tile L = Tile::of(0); L < 26; ++L) {
      for (int i = 0; i < TILE_COUNTS[L]; ++i) bag_.add(L);
    }
    for (int i = 0; i < TILE_COUNTS[BLANK]; ++i) bag_.add(BLANK);
  }

  void set_name(int player, const std::string& name) {
    if (player < 0 || player > 1) {
      status_ = "Invalid player index";
      return;
    }
    std::string n = trim_name(name);
    if (n.empty()) {
      status_ = "Name cannot be empty";
      return;
    }
    names_[player] = n;
    status_ = "";
  }

  void set_rack_slot(int player, int slot, const std::string& letter) {
    if (player < 0 || player > 1 || slot < 0 || slot >= kRackSlots) {
      status_ = "Invalid rack slot";
      return;
    }

    if (racks_[player][slot].has_value()) {
      bag_.add(racks_[player][slot].value());
      racks_[player][slot].reset();
    }

    if (letter.empty()) {
      status_ = "";
      return;
    }

    const Tile t = tile_from_letter(letter, letter == "?");
    if (t.is_empty()) {
      status_ = "Invalid tile letter";
      return;
    }
    if (!bag_.remove(t)) {
      status_ = "Tile unavailable in bag";
      return;
    }
    racks_[player][slot] = t;
    status_ = "";
  }

  void clear_rack_slot(int player, int slot) { set_rack_slot(player, slot, ""); }

  void pass_turn(int player) {
    if (player != turn_player_) {
      status_ = "It is not that player's turn";
      return;
    }
    const RackSlots before_slots = racks_[player];
    TurnRecord rec;
    rec.player = player;
    rec.rack_before = rack_known_tiles_from_slots(before_slots);
    rec.bag_size_before = bag_.size();
    rec.move = Move::pass();
    rec.score_delta = 0;
    rec.cumulative_scores = scores_;
    rec.drawn = Rack();

    ManualTurn turn;
    turn.record = rec;
    turn.include_rack_before = rack_fully_known(before_slots);
    turn.notation = "pass";
    turn.rack_before_slots = before_slots;
    turn.racks_after_turn = {racks_[0], racks_[1]};
    turns_.push_back(std::move(turn));

    turn_player_ = 1 - turn_player_;
    status_ = "";
  }

  void exchange_turn(int player, const std::vector<int>& slots) {
    if (player != turn_player_) {
      status_ = "It is not that player's turn";
      return;
    }
    if (slots.empty()) {
      status_ = "Choose at least one tile to exchange";
      return;
    }

    std::set<int> unique_slots;
    for (int slot : slots) {
      if (slot < 0 || slot >= kRackSlots || unique_slots.count(slot) > 0) {
        status_ = "Invalid exchange selection";
        return;
      }
      unique_slots.insert(slot);
    }

    const RackSlots before_slots = racks_[player];
    const int bag_size_before = bag_.size();
    TileCounts exchanged_known;
    std::string exchange_field;
    exchange_field.reserve(unique_slots.size());
    bool all_unknown = true;
    for (int slot : unique_slots) {
      if (before_slots[slot].has_value()) {
        const Tile tile = before_slots[slot].value();
        exchanged_known.add(tile);
        bag_.add(tile);
        exchange_field.push_back(tile.to_char());
        all_unknown = false;
      } else {
        exchange_field.push_back('_');
      }
      racks_[player][slot].reset();
    }
    if (all_unknown) exchange_field = std::to_string(unique_slots.size());

    TurnRecord rec;
    rec.player = player;
    rec.rack_before = rack_known_tiles_from_slots(before_slots);
    rec.bag_size_before = bag_size_before;
    rec.move = Move::exchange(exchanged_known);
    rec.score_delta = 0;
    rec.cumulative_scores = scores_;
    rec.drawn = Rack();

    ManualTurn turn;
    turn.record = rec;
    turn.include_rack_before = rack_fully_known(before_slots);
    turn.notation = "exch " + exchange_field;
    turn.rack_before_slots = before_slots;
    turn.racks_after_turn = {racks_[0], racks_[1]};
    turn.exchange_field = exchange_field;
    turns_.push_back(std::move(turn));

    turn_player_ = 1 - turn_player_;
    status_ = "";
  }

  void play_turn(int player, int row, int col, const std::string& dir, const std::string& word,
                 const boost::json::array& placements) {
    if (player != turn_player_) {
      status_ = "It is not that player's turn";
      return;
    }
    const bool horizontal = (dir == "horizontal");
    if (!horizontal && dir != "vertical") {
      status_ = "Direction must be horizontal or vertical";
      return;
    }
    if (word.empty()) {
      status_ = "Word cannot be empty";
      return;
    }

    std::vector<ManualTilePlacement> spec;
    spec.reserve(placements.size());
    TileCounts required;

    for (const boost::json::value& v : placements) {
      if (!v.is_object()) continue;
      const boost::json::object& o = v.as_object();
      ManualTilePlacement p;
      p.row = int_field(o, "row");
      p.col = int_field(o, "col");
      p.is_blank = bool_field(o, "isBlank");
      const std::string letter = str_field(o, "letter");
      p.tile = tile_from_letter(letter, p.is_blank);
      p.letter = letter.empty() ? '\0' : upper_ch(letter[0]);
      p.from_rack = str_field(o, "source") == "rack";
      p.rack_slot = int_field(o, "slot");
      if (p.row < 0 || p.col < 0 || p.row >= BOARD_SIZE || p.col >= BOARD_SIZE ||
          p.tile.is_empty() || p.letter < 'A' || p.letter > 'Z') {
        status_ = "Invalid tile placement";
        return;
      }
      required.add(p.tile);
      spec.push_back(p);
    }

    if (spec.empty()) {
      status_ = "No placed tiles were provided";
      return;
    }

    Rack gen_rack;
    for (Tile L = Tile::of(0); L < 26; ++L) {
      for (int i = 0; i < required.count(L); ++i) gen_rack.add(L);
    }
    for (int i = 0; i < required.count(BLANK); ++i) gen_rack.add(BLANK);
    while (gen_rack.size() < RACK_SIZE) gen_rack.add(Tile::of(4));  // filler 'E'

    std::vector<Move> legal = movegen_.generate(gen_rack);
    Move chosen;
    bool found = false;
    for (const Move& m : legal) {
      if (!matches_play(m, spec)) continue;
      chosen = m;
      found = true;
      break;
    }
    if (!found) {
      status_ = "No legal move matches those placements";
      return;
    }

    std::set<int> used_slots;
    TileCounts bag_needed;
    for (const ManualTilePlacement& p : spec) {
      if (p.from_rack) {
        if (p.rack_slot < 0 || p.rack_slot >= kRackSlots || used_slots.count(p.rack_slot) > 0) {
          status_ = "Invalid or repeated rack slot in play";
          return;
        }
        if (!racks_[player][p.rack_slot].has_value()) {
          status_ = "Referenced rack slot is unknown";
          return;
        }
        if (racks_[player][p.rack_slot].value() != p.tile) {
          status_ = "Rack slot tile does not match placement";
          return;
        }
        used_slots.insert(p.rack_slot);
      } else {
        bag_needed.add(p.tile);
      }
    }

    for (Tile L = Tile::of(0); L < 26; ++L) {
      if (bag_needed.count(L) > bag_.count(L)) {
        status_ = "Bag does not contain all dragged tiles";
        return;
      }
    }
    if (bag_needed.count(BLANK) > bag_.count(BLANK)) {
      status_ = "Bag does not contain all dragged tiles";
      return;
    }

    for (Tile L = Tile::of(0); L < 26; ++L) {
      for (int i = 0; i < bag_needed.count(L); ++i) bag_.remove(L);
    }
    for (int i = 0; i < bag_needed.count(BLANK); ++i) bag_.remove(BLANK);

    const RackSlots before_slots = racks_[player];
    const int bag_size_before = bag_.size();
    for (int slot : used_slots) racks_[player][slot].reset();

    const Board before = board_;
    TurnRecord rec;
    rec.player = player;
    rec.rack_before = rack_known_tiles_from_slots(before_slots);
    rec.bag_size_before = bag_size_before;
    rec.move = chosen;
    rec.score_delta = chosen.score();
    scores_[player] += chosen.score();
    rec.cumulative_scores = scores_;
    rec.drawn = Rack();

    board_.apply(chosen);

    ManualTurn t;
    t.record = rec;
    t.include_rack_before = rack_fully_known(before_slots);
    t.notation = move_to_notation(before, chosen);
    t.rack_before_slots = before_slots;
    t.racks_after_turn = {racks_[0], racks_[1]};
    turns_.push_back(std::move(t));

    turn_player_ = 1 - turn_player_;
    status_ = "";
  }

  bool export_gcg(const std::string& path) {
    std::string out_path = path;
    if (out_path.empty()) out_path = "/workspace/repo/manual_game.gcg";

    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(p.parent_path(), ec);
    }

    GameLog log;
    log.player_names = names_;
    log.final_scores = scores_;
    log.final_racks = {rack_known_tiles(0), rack_known_tiles(1)};
    log.end_reason = "manual";

    std::vector<std::optional<std::string>> rack_fields;
    std::vector<std::optional<std::string>> exchange_fields;
    std::vector<GcgWriteOptions::PostEventRacks> post_event_racks;
    rack_fields.reserve(turns_.size());
    exchange_fields.reserve(turns_.size());
    post_event_racks.reserve(turns_.size());
    for (const ManualTurn& t : turns_) {
      log.turns.push_back(t.record);
      rack_fields.push_back(gcg_rack_field(t.rack_before_slots));
      exchange_fields.push_back(t.exchange_field);
      post_event_racks.push_back(
        {maybe_rack_pragma(t.racks_after_turn[0]), maybe_rack_pragma(t.racks_after_turn[1])});
    }

    GcgWriteOptions opts;
    opts.lexicon_name = Lexicon::instance().name();
    opts.notes.push_back("Generated by Scribblez manual GCG tool on " + now_string());
    opts.initial_rack1 = maybe_rack_pragma(racks_[0]);
    opts.initial_rack2 = maybe_rack_pragma(racks_[1]);
    opts.rack_before_fields = std::move(rack_fields);
    opts.exchange_fields = std::move(exchange_fields);
    opts.post_event_racks = std::move(post_event_racks);

    std::ofstream out(out_path);
    if (!out) {
      status_ = "Failed to open export path: " + out_path;
      return false;
    }
    write_game_log_gcg(log, out, opts);
    status_ = "Exported " + out_path;
    return true;
  }

 private:
  Rack rack_known_tiles_from_slots(const RackSlots& slots) const {
    Rack r;
    for (int i = 0; i < kRackSlots; ++i) {
      if (slots[i].has_value()) r.add(slots[i].value());
    }
    return r;
  }

  Rack rack_known_tiles(int player) const { return rack_known_tiles_from_slots(racks_[player]); }

  int known_count(int player) const {
    int n = 0;
    for (int i = 0; i < kRackSlots; ++i) {
      if (racks_[player][i].has_value()) ++n;
    }
    return n;
  }

  bool rack_fully_known(int player) const { return rack_fully_known(racks_[player]); }

  bool rack_fully_known(const RackSlots& slots) const {
    for (int i = 0; i < kRackSlots; ++i) {
      if (!slots[i].has_value()) return false;
    }
    return true;
  }

  bool matches_play(const Move& m, const std::vector<ManualTilePlacement>& spec) const {
    if (m.type() != MoveType::PLAY) return false;
    if (m.num_glyphs() != static_cast<int>(spec.size())) return false;

    std::map<std::pair<int, int>, const ManualTilePlacement*> placed;
    for (const ManualTilePlacement& p : spec) {
      const auto key = std::make_pair(p.row, p.col);
      if (placed.count(key) > 0) return false;
      placed[key] = &p;
    }

    int gi = 0;

    const uint16_t mask = m.square_mask();
    for (int lane = 0; lane < BOARD_SIZE; ++lane) {
      if ((mask & (static_cast<uint16_t>(1) << lane)) == 0) continue;
      const int r = m.horizontal() ? m.start() : lane;
      const int c = m.horizontal() ? lane : m.start();
      auto it = placed.find({r, c});
      if (it == placed.end()) return false;
      if (gi >= m.num_glyphs()) return false;

      const ManualTilePlacement* p = it->second;
      Glyph g = m.glyph(gi++);
      if (g.is_blank() != p->is_blank) return false;
      if (g.letter().to_char() != p->letter) return false;
    }

    return gi == m.num_glyphs();
  }

  const Dictionary& dict_;
  Board board_;
  MoveGenerator movegen_;
  std::array<std::string, 2> names_;
  std::array<int, 2> scores_ = {0, 0};
  std::array<std::array<std::optional<Tile>, kRackSlots>, 2> racks_;
  TileCounts bag_;
  std::vector<ManualTurn> turns_;
  int turn_player_ = 0;
  std::string status_;
};

void handle_message(ManualGame& game, const boost::json::object& obj) {
  const std::string type = str_field(obj, "type");
  if (type == "set_name") {
    game.set_name(int_field(obj, "player"), str_field(obj, "name"));
    return;
  }
  if (type == "set_rack_slot") {
    game.set_rack_slot(int_field(obj, "player"), int_field(obj, "slot"), str_field(obj, "letter"));
    return;
  }
  if (type == "clear_rack_slot") {
    game.clear_rack_slot(int_field(obj, "player"), int_field(obj, "slot"));
    return;
  }
  if (type == "pass") {
    game.pass_turn(int_field(obj, "player"));
    return;
  }
  if (type == "exchange") {
    std::vector<int> slots;
    auto it = obj.find("slots");
    if (it != obj.end() && it->value().is_array()) {
      for (const boost::json::value& v : it->value().as_array()) {
        if (v.is_int64()) slots.push_back(static_cast<int>(v.as_int64()));
      }
    }
    game.exchange_turn(int_field(obj, "player"), slots);
    return;
  }
  if (type == "play") {
    boost::json::array placements;
    auto it = obj.find("placements");
    if (it != obj.end() && it->value().is_array()) placements = it->value().as_array();
    game.play_turn(int_field(obj, "player"), int_field(obj, "row"), int_field(obj, "col"),
                   str_field(obj, "dir"), str_field(obj, "word"), placements);
    return;
  }
  if (type == "export") {
    game.export_gcg(str_field(obj, "path"));
    return;
  }
  if (type == "reset") {
    game.reset();
    return;
  }
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    int ws_port = 8082;
    int vite_port = 5174;
    std::string web_dir = "web";

    po::options_description desc("manual_gcg_tool options");
    desc.add_options()("help,h", "show this help message and exit")(
      "port", po::value<int>(&ws_port)->default_value(ws_port), "engine WebSocket port")(
      "vite-port", po::value<int>(&vite_port)->default_value(vite_port), "browser UI port")(
      "web-dir", po::value<std::string>(&web_dir)->default_value(web_dir),
      "front-end package dir (cwd of npm run dev)");
    scribblez::Lexicon::instance().add_options(desc);

    scribblez::parse_command_line(argc, argv, desc);

    const scribblez::Dictionary& dict = scribblez::GameRunner::load_dictionary_or_throw();
    scribblez::WebSession session(ws_port);
    scribblez::ViteDevServer vite(web_dir, vite_port, ws_port);
    if (!vite.wait_until_ready()) {
      throw std::runtime_error("the Vite dev server did not start; see web/.vite-dev.log");
    }

    std::cerr << "\nManual GCG tool ready at " << vite.url()
              << "?tool=manual (lexicon: " << scribblez::Lexicon::instance().name() << ")\n";
    std::string cmd = "xdg-open '" + vite.url() + "?tool=manual' >/dev/null 2>&1 &";
    int rc = std::system(cmd.c_str());
    (void)rc;

    scribblez::ManualGame game(dict);

    while (true) {
      if (!session.connected()) {
        if (!session.wait_for_client()) break;
      }
      session.send_text(boost::json::serialize(game.state_json()));
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
        scribblez::handle_message(game, parsed.as_object());
        session.send_text(boost::json::serialize(game.state_json()));
      }
    }

    return 0;
  } catch (const scribblez::CleanExit&) {
    return 0;
  } catch (const scribblez::Exception&) {
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
    return 1;
  }
}
