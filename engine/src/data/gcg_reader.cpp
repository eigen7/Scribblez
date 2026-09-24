#include "data/gcg_reader.h"

#include "data/gcg_writer.h"
#include "game/tile.h"
#include "util/assert.h"
#include "util/exception.h"

#include <cctype>
#include <exception>
#include <format>
#include <istream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

// std::getline that also strips a trailing '\r'. GCG files from Windows tools
// and web exports use CRLF, and a kept '\r' would end up inside the line's
// last token (a player's name, a rack, a score).
bool getline_lf_or_crlf(std::istream& in, std::string& line) {
  if (!std::getline(in, line)) return false;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return true;
}

std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream iss(s);
  std::vector<std::string> out;
  std::string tok;
  while (iss >> tok) out.push_back(tok);
  return out;
}

std::optional<int> parse_signed_int(const std::string& tok) {
  if (tok.empty()) return std::nullopt;
  std::size_t i = 0;
  if (tok[0] == '+') i = 1;
  if (i >= tok.size()) return std::nullopt;
  try {
    return std::stoi(tok.substr(i));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool parse_gcg_position(const std::string& pos, bool* horizontal, int* row, int* col) {
  if (pos.size() < 2) return false;
  if (std::isdigit(uint8_t(pos[0])) != 0) {
    std::size_t i = 0;
    while (i < pos.size() && std::isdigit(uint8_t(pos[i])) != 0) ++i;
    if (i == 0 || i >= pos.size()) return false;
    const char c = char(std::toupper(uint8_t(pos[i])));
    if (c < 'A' || c > 'O') return false;
    const int r = std::stoi(pos.substr(0, i)) - 1;
    if (r < 0 || r >= BOARD_SIZE || i + 1 != pos.size()) return false;
    *horizontal = true;
    *row = r;
    *col = c - 'A';
    return true;
  }

  const char c = char(std::toupper(uint8_t(pos[0])));
  if (c < 'A' || c > 'O') return false;
  const std::string digits = pos.substr(1);
  if (digits.empty()) return false;
  for (char ch : digits) {
    if (std::isdigit(uint8_t(ch)) == 0) return false;
  }
  const int r = std::stoi(digits) - 1;
  if (r < 0 || r >= BOARD_SIZE) return false;
  *horizontal = false;
  *row = r;
  *col = c - 'A';
  return true;
}

// The player (0 or 1) that a "#Rack1 <tiles>" / "#Rack2 <tiles>" pragma line
// is for, or -1 if `line` is not one. Only the capitalized form that
// gcg_writer.h emits is a rack pragma: tournament GCG uses lowercase "#rack1"
// with different meaning, and the turn lines carry that information anyway.
int rack_pragma_player(const std::string& line) {
  for (int player = 0; player < 2; ++player) {
    const std::string name = std::format("#Rack{}", player + 1);
    if (line.starts_with(name) && (line.size() == name.size() || line[name.size()] == ' ')) {
      return player;
    }
  }
  return -1;
}

// The tiles neither on the snapshot's board nor known to be on a rack. A file
// that overdraws a tile leaves its count at 0: unparseable input is skipped,
// never an error.
TileCounts unaccounted_tiles(const ParsedGcgSnapshot& snapshot) {
  TileCounts tiles = TileCounts::full_distribution();
  tiles.remove(snapshot.board.tile_counts());
  for (const Rack& rack : snapshot.racks) tiles.remove(rack.counts());
  return tiles;
}

class GcgReader {
 public:
  bool Read(const std::string& gcg_text, ParsedGcgGame* out_game, std::string* error_message) {
    InitializeState();

    std::istringstream in(gcg_text);
    std::string raw;
    while (getline_lf_or_crlf(in, raw)) {
      if (raw.empty()) continue;
      if (TryParsePlayerDecl(raw)) continue;
      if (TryParseRackPragma(raw)) continue;
      if (raw[0] != '>') continue;
      ParseTurnLine(raw);
    }

    if (!saw_turn_) {
      *error_message = "No playable turns found";
      return false;
    }

    ApplyResumeRacks();
    FillBags();
    FillResult(out_game);
    return true;
  }

 private:
  void InitializeState() {
    names_ = {"Player 1", "Player 2"};
    nick_to_player_.clear();
    board_ = Board();
    scores_ = {0, 0};
    racks_ = {};
    resume_racks_ = {};
    turns_.clear();
    snapshots_.clear();
    snapshots_.push_back(CurrentSnapshot());
    end_adjustments_.clear();
    saw_turn_ = false;
  }

  bool TryParsePlayerDecl(const std::string& line) {
    if (line.rfind("#player1", 0) == 0) {
      ParsePlayerDecl(line, 0);
      return true;
    }
    if (line.rfind("#player2", 0) == 0) {
      ParsePlayerDecl(line, 1);
      return true;
    }
    return false;
  }

  // Parses a rack pragma (see rack_pragma_player). Before any event line, the
  // pragma gives a player's current rack, applied to the final position (the
  // "resume" rack). After an event, it gives their rack just after that event.
  bool TryParseRackPragma(const std::string& line) {
    const int player = rack_pragma_player(line);
    if (player < 0) return false;

    const std::size_t space = line.find(' ');
    const std::string token = space == std::string::npos ? "" : line.substr(space + 1);
    const Rack rack = RackFromToken(token);

    if (!saw_turn_) {
      resume_racks_[player] = rack;
    } else {
      racks_[player] = rack;
      snapshots_.back().racks[player] = rack;
      turns_.back().racks_after_turn[player] = rack;
    }
    return true;
  }

  void ParsePlayerDecl(const std::string& line, int player) {
    std::size_t a = line.find(' ');
    if (a == std::string::npos) return;
    std::size_t b = line.find(' ', a + 1);

    std::string nick;
    std::string name;
    if (b == std::string::npos) {
      nick = line.substr(a + 1);
      name = nick;
    } else {
      nick = line.substr(a + 1, b - (a + 1));
      name = line.substr(b + 1);
    }

    if (nick.empty()) return;
    nick_to_player_[nick] = player;
    names_[player] = name.empty() ? nick : name;
  }

  void ParseTurnLine(const std::string& line) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos || colon <= 1) return;

    const std::string nick = line.substr(1, colon - 1);
    const auto it_player = nick_to_player_.find(nick);
    if (it_player == nick_to_player_.end()) return;
    const int player = it_player->second;

    const std::string rest = line.substr(colon + 1);
    const std::vector<std::string> tok = split_ws(rest);
    if (tok.size() < 2) return;

    if (TryParseEndAdjustment(player, tok)) return;

    racks_[player] = RackFromToken(tok[0]);
    if (tok[1] == "-") {
      ParsePassTurn(player, tok);
      return;
    }
    if (!tok[1].empty() && tok[1][0] == '-') {
      ParseExchangeTurn(player, tok);
      return;
    }
    ParsePlayTurn(player, tok);
  }

  static std::string StripParens(const std::string& s) {
    std::string out;
    for (char c : s) {
      if (c != '(' && c != ')') out.push_back(c);
    }
    return out;
  }

  // Consumes the two end-of-game adjustment line shapes:
  //   ">nick: (opp_rack) +delta total"   (player went out, gains opp's tiles)
  //   ">nick: rack (rack) -delta total"  (player held tiles, penalized)
  bool TryParseEndAdjustment(int player, const std::vector<std::string>& tok) {
    if (!tok.empty() && tok.front().front() == '(') {
      if (tok.size() >= 3) RecordEndAdjustment(player, StripParens(tok[0]), tok[1], tok[2]);
      return true;
    }
    if (tok.size() >= 2 && tok[1].front() == '(') {
      if (tok.size() >= 4) RecordEndAdjustment(player, StripParens(tok[1]), tok[2], tok.back());
      return true;
    }
    return false;
  }

  void RecordEndAdjustment(int player, const std::string& tiles, const std::string& delta_tok,
                           const std::string& total_tok) {
    const auto delta = parse_signed_int(delta_tok);
    const auto total = parse_signed_int(total_tok);
    if (!delta.has_value() || !total.has_value()) return;

    ParsedGcgEndAdjustment adj;
    adj.player = player;
    adj.tiles = tiles;
    adj.delta = *delta;
    adj.total = *total;
    end_adjustments_.push_back(adj);

    // The final snapshot shows adjusted totals, not the last move's.
    scores_[player] = *total;
    if (!snapshots_.empty()) snapshots_.back().scores = scores_;
  }

  void ParsePassTurn(int player, const std::vector<std::string>& tok) {
    if (tok.size() < 4) return;
    const auto cumulative = parse_signed_int(tok.back());
    if (!cumulative.has_value()) return;

    ParsedGcgTurn turn;
    turn.record.player = player;
    turn.record.rack_before = racks_[player];
    turn.record.bag_size_before = BagSizeEstimate();
    turn.record.move = Move::pass();
    turn.record.score_delta = 0;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = "pass";
    turn.racks_after_turn = racks_;

    turns_.push_back(std::move(turn));
    snapshots_.push_back(CurrentSnapshot(1 - player));
    saw_turn_ = true;
  }

  void ParseExchangeTurn(int player, const std::vector<std::string>& tok) {
    if (tok.size() < 3) return;
    const auto cumulative = parse_signed_int(tok.back());
    if (!cumulative.has_value()) return;

    ParsedGcgTurn turn;
    turn.record.player = player;
    turn.record.rack_before = racks_[player];
    turn.record.bag_size_before = BagSizeEstimate();

    TileCounts exchanged;
    const std::string exchange_letters = tok[1].substr(1);
    for (char ch : exchange_letters) {
      const Tile t = ch == '?' ? BLANK : Tile::letter_from_char(ch);
      if (!t.is_empty()) exchanged.add(t);
    }

    turn.record.move = Move::exchange(exchanged);
    turn.record.score_delta = 0;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = "exch " + exchange_letters;
    turn.exchange_field = exchange_letters;

    racks_[player] = Rack();
    turn.racks_after_turn = racks_;

    turns_.push_back(std::move(turn));
    snapshots_.push_back(CurrentSnapshot(1 - player));
    saw_turn_ = true;
  }

  void ParsePlayTurn(int player, const std::vector<std::string>& tok) {
    if (tok.size() < 5) return;

    const std::string position = tok[1];
    const std::string word = tok[2];
    const auto score = parse_signed_int(tok[3]);
    const auto cumulative = parse_signed_int(tok[4]);
    if (!score.has_value() || !cumulative.has_value()) return;

    bool horizontal = true;
    int row = -1;
    int col = -1;
    if (!parse_gcg_position(position, &horizontal, &row, &col)) return;

    std::array<Glyph, RACK_SIZE> glyphs;
    glyphs.fill(Glyph::empty());
    int num_glyphs = 0;
    uint16_t mask = 0;
    int r = row;
    int c = col;
    bool malformed = false;

    for (char ch : word) {
      if (!board_.in_bounds(r, c)) {
        malformed = true;
        break;
      }
      if (ch == '.') {
        if (board_.at(r, c).is_empty()) {
          malformed = true;
          break;
        }
      } else {
        const Tile letter = Tile::letter_from_char(ch);
        if (letter.is_empty() || num_glyphs >= RACK_SIZE) {
          malformed = true;
          break;
        }
        const bool is_blank = std::islower(uint8_t(ch)) != 0;
        glyphs[num_glyphs++] = Glyph::played(letter, is_blank);
        const int lane = horizontal ? c : r;
        mask |= uint16_t(1) << lane;
      }
      if (horizontal) {
        ++c;
      } else {
        ++r;
      }
    }
    if (malformed) return;

    const int start = horizontal ? row : col;
    const Move move =
      Move::play(horizontal, start, mask, uint16_t(*score), glyphs.data(), num_glyphs);

    const Board before = board_;
    const int bag_size_before = BagSizeEstimate();
    board_.apply(move);

    ParsedGcgTurn turn;
    turn.record.player = player;
    turn.record.rack_before = racks_[player];
    turn.record.bag_size_before = bag_size_before;
    turn.record.move = move;
    turn.record.score_delta = *score;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = move_to_notation(before, move);

    racks_[player] = Rack();
    turn.racks_after_turn = racks_;

    turns_.push_back(std::move(turn));
    snapshots_.push_back(CurrentSnapshot(1 - player));
    saw_turn_ = true;
  }

  // The known tiles of a GCG rack field. 'A'..'Z' are tiles; '?', '*' and
  // lowercase are blanks; '_' is an unknown tile or empty slot, which takes one
  // of the RACK_SIZE slots but adds no tile; anything else ('.') is skipped.
  static Rack RackFromToken(const std::string& rack_token) {
    Rack rack;
    int slot = 0;
    for (char ch : rack_token) {
      if (slot >= RACK_SIZE) break;
      const Tile letter = Tile::letter_from_char(ch);
      const bool blank = ch == '?' || ch == '*' || (ch >= 'a' && ch <= 'z');
      if (!blank && ch != '_' && letter.is_empty()) continue;
      ++slot;
      if (ch != '_') rack.add(blank ? BLANK : letter);
    }
    return rack;
  }

  void ApplyResumeRacks() {
    for (int p = 0; p < 2; ++p) {
      if (resume_racks_[p].has_value()) snapshots_.back().racks[p] = *resume_racks_[p];
    }
  }

  // Last, since rack pragmas revise a snapshot's racks after it is taken.
  void FillBags() {
    for (ParsedGcgSnapshot& snapshot : snapshots_) snapshot.bag = unaccounted_tiles(snapshot);
  }

  // A turn line's rack may be partly hidden, but racks are full while the
  // bag holds tiles.
  int BagSizeEstimate() const { return board_.pov_bag_size(RACK_SIZE); }

  ParsedGcgSnapshot CurrentSnapshot(int turn_player = 0) const {
    ParsedGcgSnapshot snapshot;
    snapshot.board = board_;
    snapshot.scores = scores_;
    snapshot.racks = racks_;
    snapshot.turn_player = turn_player;
    return snapshot;
  }

  void FillResult(ParsedGcgGame* out_game) {
    out_game->player_names = names_;
    out_game->turns = turns_;
    out_game->snapshots = snapshots_;
    out_game->end_adjustments = end_adjustments_;
    out_game->header_racks = resume_racks_;
  }

  std::array<std::string, 2> names_ = {"Player 1", "Player 2"};
  std::map<std::string, int> nick_to_player_;
  Board board_;
  std::array<int, 2> scores_ = {0, 0};
  std::array<Rack, 2> racks_;
  std::array<std::optional<Rack>, 2> resume_racks_;
  std::vector<ParsedGcgTurn> turns_;
  std::vector<ParsedGcgSnapshot> snapshots_;
  std::vector<ParsedGcgEndAdjustment> end_adjustments_;
  bool saw_turn_ = false;
};

}  // namespace

GameLogStorage ParsedGcgGame::to_game_log_storage() const {
  GameLogStorage storage;
  storage.player_names = player_names;
  storage.turns.reserve(turns.size());
  for (const ParsedGcgTurn& turn : turns) {
    storage.turns.push_back(turn.record);
  }

  if (!snapshots.empty()) {
    storage.final_scores = snapshots.back().scores;
    storage.final_racks = snapshots.back().racks;
  }

  return storage;
}

bool read_gcg_text(const std::string& gcg_text, ParsedGcgGame* out_game,
                   std::string* error_message) {
  if (out_game == nullptr || error_message == nullptr) return false;
  GcgReader reader;
  if (!reader.Read(gcg_text, out_game, error_message)) return false;
  out_game->game_log = out_game->to_game_log_storage();
  return true;
}

Rack retained_leave(const ParsedGcgGame& game, int player) {
  for (auto it = game.turns.rbegin(); it != game.turns.rend(); ++it) {
    if (it->record.player != player) continue;
    Rack leave = it->record.rack_before;
    const Move& m = it->record.move;
    for (int i = 0; i < m.num_glyphs(); ++i) {
      const bool ok = leave.remove(m.glyph(i).rack_tile());
      RELEASE_ASSERT(ok);
    }
    return leave;
  }
  return Rack{};
}

namespace {

// The final recorded state, the side to move, and its pragma rack: the common
// start of read_gcg_endgame and read_gcg_position.
bool final_state(const std::string& gcg_text, ParsedGcgGame* game,
                 const ParsedGcgSnapshot** snapshot, int* mover, Rack* mover_rack,
                 std::string* error_message) {
  if (!read_gcg_text(gcg_text, game, error_message)) return false;
  if (game->snapshots.empty()) {
    *error_message = "GCG contains no positions";
    return false;
  }
  *snapshot = &game->snapshots.back();
  *mover = (*snapshot)->turn_player;
  const std::optional<Rack>& rack = game->header_racks[*mover];
  if (!rack.has_value()) {
    *error_message = std::format("the mover's rack is unknown: add a #Rack{} pragma", *mover + 1);
    return false;
  }
  *mover_rack = *rack;
  return true;
}

}  // namespace

bool read_gcg_endgame(const std::string& gcg_text, ParsedGcgEndgame* out,
                      std::string* error_message) {
  ParsedGcgGame game;
  const ParsedGcgSnapshot* snapshot_ptr;
  int mover;
  Rack mover_rack;
  if (!final_state(gcg_text, &game, &snapshot_ptr, &mover, &mover_rack, error_message)) {
    return false;
  }
  const ParsedGcgSnapshot& snapshot = *snapshot_ptr;
  std::optional<Rack> opp_rack = game.header_racks[1 - mover];
  if (!opp_rack.has_value()) {
    try {
      opp_rack = snapshot.board.hidden_rack(mover_rack);
    } catch (const util::Exception& e) {
      *error_message = e.what();
      return false;
    }
  }

  out->board = snapshot.board;
  out->racks[mover] = mover_rack;
  out->racks[1 - mover] = *opp_rack;
  out->scores = snapshot.scores;
  out->mover = mover;
  out->player_names = game.player_names;
  out->turns = game.turns.size();
  return true;
}

namespace {

// Fills in everything in `out` but `game`, which the caller has already set
// (cut to the moves before the position).
void lift_position(const ParsedGcgSnapshot& snapshot, int mover, const Rack& rack, bool open_leaves,
                   ParsedGcgPosition* out) {
  out->board = snapshot.board;
  out->scores = snapshot.scores;
  out->mover = mover;
  out->rack = rack;
  out->opp_leave = open_leaves ? retained_leave(out->game, 1 - mover) : Rack{};
  out->turns = out->game.turns.size();
  out->bag_size = out->board.pov_bag_size(out->rack.size());
}

}  // namespace

bool read_gcg_position(const std::string& gcg_text, bool open_leaves, ParsedGcgPosition* out,
                       std::string* error_message) {
  const ParsedGcgSnapshot* snapshot;
  int mover;
  Rack rack;
  if (!final_state(gcg_text, &out->game, &snapshot, &mover, &rack, error_message)) return false;
  lift_position(*snapshot, mover, rack, open_leaves, out);
  return true;
}

bool read_gcg_position_at(const std::string& gcg_text, int turn_index, bool open_leaves,
                          ParsedGcgPosition* out, std::string* error_message) {
  ParsedGcgGame& game = out->game;
  if (!read_gcg_text(gcg_text, &game, error_message)) return false;
  const int turns = game.turns.size();
  if (turn_index < 0 || turn_index >= turns) {
    *error_message =
      std::format("turn index {} is out of range: the GCG records {} turns", turn_index, turns);
    return false;
  }
  const TurnRecord record = game.turns[size_t(turn_index)].record;
  // Keep only the moves that lead to the position, and rebuild the log to match.
  game.turns.resize(size_t(turn_index));
  game.snapshots.resize(size_t(turn_index) + 1);
  game.game_log = game.to_game_log_storage();
  lift_position(game.snapshots.back(), record.player, record.rack_before, open_leaves, out);
  return true;
}

}  // namespace scribblez
