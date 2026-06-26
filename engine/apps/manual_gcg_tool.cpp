#include "scribblez/cli.h"
#include "scribblez/exception.h"
#include "scribblez/game.h"
#include "scribblez/game_runner.h"
#include "scribblez/gcg_reader.h"
#include "scribblez/gcg_writer.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/instance_ports.h"
#include "scribblez/lexicon.h"
#include "scribblez/macondo_bot.h"
#include "scribblez/movegen.h"
#include "scribblez/web_server.h"
#include "util/encoding.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
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

enum class RackSlotState : uint8_t { UNKNOWN, KNOWN, EMPTY };

// A rack slot as shown in the UI: KNOWN carries a revealed tile; UNKNOWN is a
// tile that is present but hidden (rendered as "?", still selectable for
// exchange); EMPTY is a slot holding no tile (rendered as empty space).
struct DisplaySlot {
  RackSlotState state = RackSlotState::UNKNOWN;
  Tile tile = EMPTY_SQUARE;
};
using RackDisplay = std::array<DisplaySlot, kRackSlots>;

// A rack's slots for display. A revealed tile is KNOWN; every other slot takes
// `hidden`: UNKNOWN ("?") when the rack is assumed full but its tiles aren't
// known, or EMPTY when the player is known to hold only their revealed tiles.
RackDisplay display_from_slots(const RackSlots& slots, RackSlotState hidden) {
  RackDisplay display;
  for (int i = 0; i < kRackSlots; ++i) {
    if (slots[i].has_value()) {
      display[i] = {RackSlotState::KNOWN, slots[i].value()};
    } else {
      display[i].state = hidden;
    }
  }
  return display;
}

struct ManualTurn {
  TurnRecord record;
  bool include_rack_before = false;
  std::string notation;
  RackSlots rack_before_slots;
  std::array<RackSlots, 2> racks_after_turn;
  std::optional<std::string> exchange_field;
};

struct ManualSnapshot {
  Board board;
  std::array<int, 2> scores = {0, 0};
  std::array<RackSlots, 2> racks;
  TileCounts bag;
  int turn_player = 0;
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

// Tiles left in the bag, estimated from the board the same way the UI's bag
// count is: 100 tiles total, less those on the board and the (up to) 14 on the
// two racks. Used to decide whether unknown rack slots are still drawable "?"
// tiles or genuinely empty.
int bag_estimate(const Board& board) { return std::max(0, 100 - board_tile_count(board) - 14); }

// Both players' racks as a JSON array of per-slot tiles. A known slot carries
// its letter/score; an unknown ("present" but hidden) slot renders as "?" and
// stays selectable for exchange; an absent slot (present:false) renders as
// empty space.
boost::json::array racks_json(const std::array<RackDisplay, 2>& display_racks) {
  boost::json::array racks;
  for (int p = 0; p < 2; ++p) {
    boost::json::array r;
    for (int s = 0; s < kRackSlots; ++s) {
      const DisplaySlot& slot = display_racks[p][s];
      switch (slot.state) {
        case RackSlotState::KNOWN:
          r.emplace_back(boost::json::object{{"letter", std::string(1, slot.tile.to_char())},
                                             {"score", slot.tile.value()},
                                             {"known", true},
                                             {"present", true}});
          break;
        case RackSlotState::UNKNOWN:
          r.emplace_back(boost::json::object{
            {"letter", "?"}, {"score", 0}, {"known", false}, {"present", true}});
          break;
        case RackSlotState::EMPTY:
          r.emplace_back(boost::json::object{
            {"letter", ""}, {"score", 0}, {"known", false}, {"present", false}});
          break;
      }
    }
    racks.emplace_back(std::move(r));
  }
  return racks;
}

// Remaining bag contents as a JSON array of {letter, score, count}, listing each
// present letter (and the blank as "?") with a positive count.
boost::json::array bag_tiles_json(const ManualSnapshot& snap) {
  boost::json::array bag_tiles;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    const int count = snap.bag.count(L);
    if (count <= 0) continue;
    bag_tiles.emplace_back(boost::json::object{
      {"letter", std::string(1, L.to_char())}, {"score", L.value()}, {"count", count}});
  }
  const int blanks = snap.bag.count(BLANK);
  if (blanks > 0) {
    bag_tiles.emplace_back(boost::json::object{{"letter", "?"}, {"score", 0}, {"count", blanks}});
  }
  return bag_tiles;
}

// The move history as a JSON array, one {player, text, score, cumulative, rack}
// entry per turn.
boost::json::array turns_json(const std::vector<ManualTurn>& turns) {
  boost::json::array turn_list;
  for (const ManualTurn& t : turns) {
    turn_list.emplace_back(boost::json::object{
      {"player", t.record.player},
      {"text", t.notation},
      {"score", t.record.score_delta},
      {"cumulative", {t.record.cumulative_scores[0], t.record.cumulative_scores[1]}},
      {"rack", t.include_rack_before ? boost::json::value(t.record.rack_before.to_string())
                                     : boost::json::value(nullptr)}});
  }
  return turn_list;
}

class ManualGame {
 public:
  explicit ManualGame(const Dictionary& dict) : dict_(dict), movegen_(board_, dict_) {
    names_[0] = "Player 1";
    names_[1] = "Player 2";
    reset();
  }

  boost::json::object state_json() const {
    const ManualSnapshot& snap = snapshots_[view_ply_];
    const int bag_count = bag_estimate(snap.board);
    const std::array<RackDisplay, 2> display_racks = display_racks_for_view(view_ply_, snap.racks);

    boost::json::object o;
    o["type"] = "manual_state";
    o["board"] = board_grid(snap.board);
    o["bonuses"] = bonus_grid(snap.board);
    o["scores"] = {snap.scores[0], snap.scores[1]};
    o["player_names"] = {names_[0], names_[1]};
    o["current_player"] = snap.turn_player;
    o["bag_count"] = bag_count;
    o["rack_known_counts"] = {known_count(display_racks[0]), known_count(display_racks[1])};
    o["lexicon"] = Lexicon::instance().name();
    o["max_name_len"] = kMaxNameLen;
    o["status"] = status_;
    o["default_export_path"] = "/workspace/repo/manual_game.gcg";
    o["tile_scores"] = tile_score_map();
    o["view_ply"] = view_ply_;
    o["tail_ply"] = static_cast<int>(turns_.size());
    o["backtracking"] = is_backtracking();
    o["game_over"] = !end_adjustments_.empty();
    o["racks"] = racks_json(display_racks);
    o["bag_tiles"] = bag_tiles_json(snap);
    o["turns"] = turns_json(turns_);
    o["end_adjustments"] = end_adjustments_json();
    o["last_move"] = last_move_squares();

    return o;
  }

  // Board squares of the tiles newly placed by the move that produced the
  // currently viewed position, as [row, col] pairs. Empty at the start position
  // and for pass/exchange turns (which place no tiles).
  boost::json::array last_move_squares() const {
    boost::json::array squares;
    if (view_ply_ <= 0 || view_ply_ > static_cast<int>(turns_.size())) return squares;
    const Move& m = turns_[view_ply_ - 1].record.move;
    if (m.type() != MoveType::PLAY) return squares;

    const uint16_t mask = m.square_mask();
    for (int lane = 0; lane < BOARD_SIZE; ++lane) {
      if ((mask & (static_cast<uint16_t>(1) << lane)) == 0) continue;
      const int r = m.horizontal() ? m.start() : lane;
      const int c = m.horizontal() ? lane : m.start();
      squares.emplace_back(boost::json::array{r, c});
    }
    return squares;
  }

  // End-of-game rack adjustments as a JSON array of {player, tiles, delta,
  // total}, one per scoring/penalty line; empty during a game in progress.
  boost::json::array end_adjustments_json() const {
    boost::json::array out;
    for (const ParsedGcgEndAdjustment& adj : end_adjustments_) {
      out.emplace_back(boost::json::object{
        {"player", adj.player}, {"tiles", adj.tiles}, {"delta", adj.delta}, {"total", adj.total}});
    }
    return out;
  }

  void reset() {
    board_ = Board();
    scores_ = {0, 0};
    turn_player_ = 0;
    turns_.clear();
    end_adjustments_.clear();
    status_ = "";
    for (int p = 0; p < 2; ++p) {
      for (int s = 0; s < kRackSlots; ++s) racks_[p][s].reset();
    }
    bag_ = TileCounts();
    for (Tile L = Tile::of(0); L < 26; ++L) {
      for (int i = 0; i < TILE_COUNTS[L]; ++i) bag_.add(L);
    }
    for (int i = 0; i < TILE_COUNTS[BLANK]; ++i) bag_.add(BLANK);

    snapshots_.clear();
    snapshots_.push_back(snapshot_from_live());
    view_ply_ = 0;
  }

  void set_name(int player, const std::string& name) {
    if (!require_live_mode("edit names")) return;
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
    if (!require_live_mode("edit racks")) return;
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
    if (!require_live_mode("play turns")) return;
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
    snapshots_.push_back(snapshot_from_live());
    view_ply_ = static_cast<int>(turns_.size());
  }

  void exchange_turn(int player, const std::vector<int>& slots) {
    if (!require_live_mode("play turns")) return;
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
    snapshots_.push_back(snapshot_from_live());
    view_ply_ = static_cast<int>(turns_.size());
  }

  void play_turn(int player, int row, int col, const std::string& dir, const std::string& word,
                 const boost::json::array& placements) {
    if (!require_live_mode("play turns")) return;
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
    snapshots_.push_back(snapshot_from_live());
    view_ply_ = static_cast<int>(turns_.size());
  }

  void jump_to_ply(int ply) {
    if (ply < 0 || ply > static_cast<int>(turns_.size())) {
      status_ = "Invalid history position";
      return;
    }
    view_ply_ = ply;
    if (is_backtracking()) {
      status_ = "Viewing turn " + std::to_string(view_ply_) + " / " +
                std::to_string(static_cast<int>(turns_.size()));
    } else {
      status_.clear();
    }
  }

  void fork_game() {
    if (!is_backtracking()) {
      status_ = "Already at latest position";
      return;
    }
    const std::array<RackDisplay, 2> fork_racks =
      display_racks_for_view(view_ply_, snapshots_[view_ply_].racks);
    end_adjustments_.clear();
    turns_.resize(view_ply_);
    snapshots_.resize(static_cast<std::size_t>(view_ply_ + 1));
    restore_live_from_snapshot(snapshots_.back());
    for (int p = 0; p < 2; ++p) racks_[p] = slots_from_display(fork_racks[p]);
    snapshots_.back().racks = racks_;
    view_ply_ = static_cast<int>(turns_.size());
    status_ = "Forked game at turn " + std::to_string(view_ply_);
  }

  // Play a full game between two in-process HastyBot agents and load the result
  // as the current game, so the whole self-play game can be reviewed turn by
  // turn. The played-out game log is serialized to GCG and routed through the
  // same load path as an imported file, reusing its snapshot/rack
  // reconstruction.
  void create_random_game() {
    HastyEquity::ensure_initialized(Lexicon::instance().name());
    HastyBotAgent player0(0, "Hasty 1");
    HastyBotAgent player1(0, "Hasty 2");

    std::random_device rd;
    const uint64_t seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    Game game(player0, player1, dict_, seed);
    game.play();

    load_gcg_text(game_log_to_gcg(game.log()), "random hasty-vs-hasty game");
    if (!turns_.empty()) {
      status_ = "Created random game: " + names_[0] + " " + std::to_string(scores_[0]) + " - " +
                std::to_string(scores_[1]) + " " + names_[1];
    }
  }

  void load_gcg_text(const std::string& gcg_text, const std::string& source_name) {
    reset();

    ParsedGcgGame parsed;
    std::string error;
    if (!read_gcg_text(gcg_text, &parsed, &error)) {
      status_ = error + " in " + source_name;
      return;
    }

    names_ = parsed.player_names;
    turns_.clear();
    turns_.reserve(parsed.turns.size());
    for (int i = 0; i < static_cast<int>(parsed.turns.size()); ++i) {
      const ParsedGcgTurn& parsed_turn = parsed.turns[i];
      ManualTurn turn;
      turn.record = parsed.game_log.turns[i];
      turn.include_rack_before = rack_fully_known(parsed_turn.rack_before_slots);
      turn.notation = parsed_turn.notation;
      turn.rack_before_slots = parsed_turn.rack_before_slots;
      turn.racks_after_turn = parsed_turn.racks_after_turn;
      turn.exchange_field = parsed_turn.exchange_field;
      turns_.push_back(std::move(turn));
    }

    snapshots_.clear();
    snapshots_.reserve(parsed.snapshots.size());
    for (const ParsedGcgSnapshot& parsed_snapshot : parsed.snapshots) {
      ManualSnapshot snapshot;
      snapshot.board = parsed_snapshot.board;
      snapshot.scores = parsed_snapshot.scores;
      snapshot.racks = parsed_snapshot.racks;
      snapshot.bag = parsed_snapshot.bag;
      snapshot.turn_player = parsed_snapshot.turn_player;
      snapshots_.push_back(std::move(snapshot));
    }

    end_adjustments_ = parsed.end_adjustments;

    restore_live_from_snapshot(snapshots_.back());
    view_ply_ = static_cast<int>(turns_.size());
    status_ = "Loaded " + source_name;
  }

  bool export_gcg(const std::string& path) {
    std::string out_path = path;
    if (out_path.empty()) out_path = "/workspace/repo/manual_game.gcg";

    std::filesystem::path p(out_path);
    if (p.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(p.parent_path(), ec);
    }

    GameLogStorage log;
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
    write_game_log_gcg(log.view(), out, opts);
    status_ = "Exported " + out_path;
    return true;
  }

  // Write a PNG rendered by the browser (base64, no data-URL prefix) to `path`.
  void export_png(const std::string& path, const std::string& base64_data) {
    if (path.empty()) {
      status_ = "No export path given";
      return;
    }
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::error_code ec;
      std::filesystem::create_directories(p.parent_path(), ec);
    }
    const std::string bytes = util::base64_decode(base64_data);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      status_ = "Failed to open export path: " + path;
      return;
    }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    status_ = "Exported " + path;
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

  int known_count(int player) const { return known_count(racks_[player]); }

  int known_count(const RackSlots& slots) const {
    int n = 0;
    for (int i = 0; i < kRackSlots; ++i) {
      if (slots[i].has_value()) ++n;
    }
    return n;
  }

  int known_count(const RackDisplay& display) const {
    int n = 0;
    for (const DisplaySlot& slot : display) {
      if (slot.state == RackSlotState::KNOWN) ++n;
    }
    return n;
  }

  // The known tiles of a display rack as plain rack slots; unknown and absent
  // slots become unset. Used when forking a game off a viewed position.
  RackSlots slots_from_display(const RackDisplay& display) const {
    RackSlots slots;
    for (int i = 0; i < kRackSlots; ++i) {
      if (display[i].state == RackSlotState::KNOWN) slots[i] = display[i].tile;
    }
    return slots;
  }

  // The leftover tiles of `player` as recorded in an end-of-game adjustment, if
  // any. A gain line (delta >= 0) scores the opponent's tiles, so its tiles
  // belong to the other player; a penalty line (delta < 0) scores the player's
  // own tiles.
  std::optional<std::string> end_rack_tiles_for(int player) const {
    for (const ParsedGcgEndAdjustment& adj : end_adjustments_) {
      const int owner = adj.delta >= 0 ? 1 - adj.player : adj.player;
      if (owner == player && !adj.tiles.empty()) return adj.tiles;
    }
    return std::nullopt;
  }

  // Parse a rack string ("AER?" etc.; '?' is a blank) into rack slots.
  RackSlots rack_slots_from_letters(const std::string& letters) const {
    RackSlots slots;
    int i = 0;
    for (char ch : letters) {
      if (i >= kRackSlots) break;
      const char up = upper_ch(ch);
      if (up == '?') {
        slots[i++] = BLANK;
      } else if (up >= 'A' && up <= 'Z') {
        slots[i++] = Tile::from_char(up);
      }
    }
    return slots;
  }

  bool rack_fully_known(int player) const { return rack_fully_known(racks_[player]); }

  bool rack_fully_known(const RackSlots& slots) const {
    for (int i = 0; i < kRackSlots; ++i) {
      if (!slots[i].has_value()) return false;
    }
    return true;
  }

  RackSlots next_known_rack_before(int player, int from_ply, const RackSlots& fallback) const {
    for (int i = from_ply; i < static_cast<int>(turns_.size()); ++i) {
      const ManualTurn& t = turns_[i];
      if (t.record.player != player) continue;
      if (has_known_tiles(t.rack_before_slots)) return t.rack_before_slots;
    }
    return fallback;
  }

  RackDisplay post_move_rack_from_turn(const ManualTurn& t) const {
    std::array<RackSlotState, kRackSlots> state;
    std::array<std::optional<Tile>, kRackSlots> letters;
    for (int i = 0; i < kRackSlots; ++i) {
      if (t.rack_before_slots[i].has_value()) {
        state[i] = RackSlotState::KNOWN;
        letters[i] = t.rack_before_slots[i].value();
      } else {
        state[i] = RackSlotState::UNKNOWN;
        letters[i].reset();
      }
    }

    if (t.record.move.type() == MoveType::PLAY || t.record.move.type() == MoveType::EXCHANGE) {
      for (int i = 0; i < t.record.move.num_glyphs(); ++i) {
        consume_rack_tile_for_post_move(&state, &letters, t.record.move.glyph(i).rack_tile());
      }
    }

    // Show the player's leave: the tiles still in hand after the move. Played
    // tiles leave their slots empty, and drawn replacements are not displayed
    // (they render as empty space too). Remaining known tiles render as
    // themselves; a remaining hidden tile renders as "?".
    RackDisplay out;
    for (int i = 0; i < kRackSlots; ++i) {
      out[i].state = state[i];
      if (state[i] == RackSlotState::KNOWN) out[i].tile = letters[i].value();
    }
    return out;
  }

  void consume_rack_tile_for_post_move(std::array<RackSlotState, kRackSlots>* state,
                                       std::array<std::optional<Tile>, kRackSlots>* letters,
                                       Tile rack_tile) const {
    for (int i = 0; i < kRackSlots; ++i) {
      if ((*state)[i] == RackSlotState::KNOWN && (*letters)[i].has_value() &&
          (*letters)[i].value() == rack_tile) {
        (*state)[i] = RackSlotState::EMPTY;
        (*letters)[i].reset();
        return;
      }
    }
    for (int i = 0; i < kRackSlots; ++i) {
      if ((*state)[i] == RackSlotState::UNKNOWN) {
        (*state)[i] = RackSlotState::EMPTY;
        return;
      }
    }
  }

  std::array<RackDisplay, 2> display_racks_for_view(
    int view_ply, const std::array<RackSlots, 2>& fallback) const {
    // An unrevealed slot of a rack we have no count for is assumed to be a held
    // (present) tile, shown as "?". A rack might legitimately be full and hidden
    // even at the end of the game (e.g. the opponent's rack after the final
    // play), so empty slots are not inferred from the bag being empty.
    std::array<RackDisplay, 2> out = {display_from_slots(fallback[0], RackSlotState::UNKNOWN),
                                      display_from_slots(fallback[1], RackSlotState::UNKNOWN)};
    if (view_ply <= 0 || view_ply > static_cast<int>(turns_.size())) return out;

    const ManualTurn& viewed = turns_[view_ply - 1];
    const int mover = viewed.record.player;
    const int waiting = 1 - mover;

    out[mover] = post_move_rack_from_turn(viewed);
    out[waiting] = display_from_slots(next_known_rack_before(waiting, view_ply, fallback[waiting]),
                                      RackSlotState::UNKNOWN);

    // At the final position, the player who didn't go out still holds their
    // leftover tiles. These survive only in the end-of-game adjustment (the
    // per-turn racks are cleared after each move), so reveal them here. The
    // adjustment lists their whole rack, so any other slot is genuinely empty.
    if (view_ply == static_cast<int>(turns_.size())) {
      if (const auto tiles = end_rack_tiles_for(waiting)) {
        out[waiting] = display_from_slots(rack_slots_from_letters(*tiles), RackSlotState::EMPTY);
      }
    }
    return out;
  }

  ManualSnapshot snapshot_from_live() const {
    ManualSnapshot s;
    s.board = board_;
    s.scores = scores_;
    s.racks = racks_;
    s.bag = bag_;
    s.turn_player = turn_player_;
    return s;
  }

  void restore_live_from_snapshot(const ManualSnapshot& s) {
    board_ = s.board;
    scores_ = s.scores;
    racks_ = s.racks;
    bag_ = s.bag;
    turn_player_ = s.turn_player;
  }

  bool is_backtracking() const { return view_ply_ != static_cast<int>(turns_.size()); }

  bool require_live_mode(const std::string& action) {
    if (!is_backtracking()) return true;
    status_ = "Cannot " + action + " while viewing history. Return to the latest turn or fork.";
    return false;
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
  std::vector<ManualSnapshot> snapshots_;
  std::vector<ParsedGcgEndAdjustment> end_adjustments_;
  int view_ply_ = 0;
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
  if (type == "jump_to_ply") {
    game.jump_to_ply(int_field(obj, "ply"));
    return;
  }
  if (type == "fork_game") {
    game.fork_game();
    return;
  }
  if (type == "load_gcg_text") {
    game.load_gcg_text(str_field(obj, "text"), str_field(obj, "file_name"));
    return;
  }
  if (type == "export") {
    game.export_gcg(str_field(obj, "path"));
    return;
  }
  if (type == "export_png") {
    game.export_png(str_field(obj, "path"), str_field(obj, "data"));
    return;
  }
  if (type == "reset") {
    game.reset();
    return;
  }
  if (type == "create_random_game") {
    game.create_random_game();
    return;
  }
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    int ws_port = 8082 + scribblez::instance_port_offset();
    int vite_port = 5174 + scribblez::instance_port_offset();
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
    scribblez::ViteDevServer vite(web_dir, vite_port, ws_port, "manual");
    if (!vite.wait_until_ready()) {
      throw std::runtime_error("the Vite dev server did not start; see web/.vite-dev.log");
    }

    std::cerr << "\nManual GCG tool ready at " << vite.url()
              << " (lexicon: " << scribblez::Lexicon::instance().name() << ")\n";
    std::string cmd = "xdg-open '" + vite.url() + "' >/dev/null 2>&1 &";
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
