#include "data/gcg_reader.h"

#include "game/tile.h"
#include "serve/web_server.h"
#include "util/assert.h"
#include "util/exception.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <format>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

char upper_ch(char c) {
  if (c >= 'a' && c <= 'z') return char(c - 'a' + 'A');
  return c;
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
    const char c = upper_ch(pos[i]);
    if (c < 'A' || c > 'O') return false;
    const int r = std::stoi(pos.substr(0, i)) - 1;
    if (r < 0 || r >= BOARD_SIZE || i + 1 != pos.size()) return false;
    *horizontal = true;
    *row = r;
    *col = c - 'A';
    return true;
  }

  const char c = upper_ch(pos[0]);
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

int board_tile_count(const Board& board) {
  int n = 0;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!board.at(r, c).is_empty()) ++n;
    }
  }
  return n;
}

TileCounts full_bag() {
  TileCounts bag;
  for (Tile l = Tile::of(0); l < 26; ++l) {
    for (int i = 0; i < TILE_COUNTS[l]; ++i) bag.add(l);
  }
  for (int i = 0; i < TILE_COUNTS[BLANK]; ++i) bag.add(BLANK);
  return bag;
}

class GcgReader {
 public:
  bool Read(const std::string& gcg_text, ParsedGcgGame* out_game, std::string* error_message) {
    InitializeState();

    std::istringstream in(gcg_text);
    std::string raw;
    while (std::getline(in, raw)) {
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
    FillResult(out_game);
    return true;
  }

 private:
  void InitializeState() {
    names_ = {"Player 1", "Player 2"};
    nick_to_player_.clear();
    board_ = Board();
    scores_ = {0, 0};
    for (int p = 0; p < 2; ++p) {
      ClearRackSlots(p);
      resume_racks_[p].reset();
    }
    bag_ = full_bag();
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

  // Parse the manual tool's own rack pragmata, "#Rack1 <tiles>" / "#Rack2
  // <tiles>". Before any event line the pragma gives a player's current
  // ("resume") rack, held for the final position; after an event it gives their
  // rack just after that event. Standard tournament GCG spells the starting
  // rack in lowercase ("#rack1"), which carries different semantics and is left
  // for the turn lines to supply, so only the capitalized form is consumed here.
  bool TryParseRackPragma(const std::string& line) {
    int player = 0;
    if (line.rfind("#Rack1", 0) == 0) {
      player = 0;
    } else if (line.rfind("#Rack2", 0) == 0) {
      player = 1;
    } else {
      return false;
    }

    const std::size_t space = line.find(' ');
    const std::string token = space == std::string::npos ? "" : line.substr(space + 1);
    const ParsedRackSlots slots = SlotsFromRackToken(token);

    if (!saw_turn_) {
      resume_racks_[player] = slots;
    } else {
      racks_[player] = slots;
      snapshots_.back().racks[player] = slots;
      turns_.back().racks_after_turn[player] = slots;
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

    SetRackSlotsFromToken(player, tok[0]);
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

  // Recognize the two end-of-game adjustment line shapes and record them
  // instead of treating them as plays:
  //   ">nick: (opp_rack) +delta total"        (player went out, gains tiles)
  //   ">nick: rack (rack) -delta total"        (player held tiles, penalized)
  // Returns true if the line was an adjustment (and was consumed).
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

    // Fold the adjustment into the final scores so the end-of-game position
    // shows the adjusted totals, not the last move's cumulative.
    scores_[player] = *total;
    if (!snapshots_.empty()) snapshots_.back().scores = scores_;
  }

  void ParsePassTurn(int player, const std::vector<std::string>& tok) {
    if (tok.size() < 4) return;
    const auto cumulative = parse_signed_int(tok.back());
    if (!cumulative.has_value()) return;

    ParsedGcgTurn turn;
    turn.rack_before_slots = racks_[player];
    turn.record.player = player;
    turn.record.rack_before = RackFromSlots(turn.rack_before_slots);
    turn.record.bag_size_before = BagSizeEstimate();
    turn.record.move = Move::pass();
    turn.record.score_delta = 0;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = "pass";
    turn.racks_after_turn = {racks_[0], racks_[1]};

    turns_.push_back(std::move(turn));
    snapshots_.push_back(CurrentSnapshot(1 - player));
    saw_turn_ = true;
  }

  void ParseExchangeTurn(int player, const std::vector<std::string>& tok) {
    if (tok.size() < 3) return;
    const auto cumulative = parse_signed_int(tok.back());
    if (!cumulative.has_value()) return;

    ParsedGcgTurn turn;
    turn.rack_before_slots = racks_[player];
    turn.record.player = player;
    turn.record.rack_before = RackFromSlots(turn.rack_before_slots);
    turn.record.bag_size_before = BagSizeEstimate();

    TileCounts exchanged;
    const std::string exchange_letters = tok[1].substr(1);
    for (char ch : exchange_letters) {
      const char up = upper_ch(ch);
      if (up == '?') {
        exchanged.add(BLANK);
      } else if (up >= 'A' && up <= 'Z') {
        exchanged.add(Tile::from_char(up));
      }
    }

    turn.record.move = Move::exchange(exchanged);
    turn.record.score_delta = 0;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = "exch " + exchange_letters;
    turn.exchange_field = exchange_letters;

    ClearRackSlots(player);
    turn.racks_after_turn = {racks_[0], racks_[1]};

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
        const char up = upper_ch(ch);
        if (up < 'A' || up > 'Z' || num_glyphs >= RACK_SIZE) {
          malformed = true;
          break;
        }
        const bool is_blank = std::islower(uint8_t(ch)) != 0;
        glyphs[num_glyphs++] = Glyph::played(Tile::from_char(up), is_blank);
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
    turn.rack_before_slots = racks_[player];
    turn.record.player = player;
    turn.record.rack_before = RackFromSlots(turn.rack_before_slots);
    turn.record.bag_size_before = bag_size_before;
    turn.record.move = move;
    turn.record.score_delta = *score;
    scores_[player] = *cumulative;
    turn.record.cumulative_scores = scores_;
    turn.notation = move_to_notation(before, move);

    ClearRackSlots(player);
    turn.racks_after_turn = {racks_[0], racks_[1]};

    turns_.push_back(std::move(turn));
    snapshots_.push_back(CurrentSnapshot(1 - player));
    saw_turn_ = true;
  }

  void ClearRackSlots(int player) {
    for (int i = 0; i < RACK_SIZE; ++i) racks_[player][i].reset();
  }

  // Parse a rack token into slots: 'A'..'Z' are tiles, '?'/'*'/lowercase are
  // blanks, '_' marks a present-but-unknown slot, '.' is skipped.
  static ParsedRackSlots SlotsFromRackToken(const std::string& rack_token) {
    ParsedRackSlots slots;
    int slot = 0;
    for (char ch : rack_token) {
      if (slot >= RACK_SIZE) break;
      if (ch == '.') continue;
      const char up = upper_ch(ch);
      if (up == '_') {
        ++slot;
        continue;
      }
      if (up == '?' || ch == '*' || (ch >= 'a' && ch <= 'z')) {
        slots[slot++] = BLANK;
        continue;
      }
      if (up >= 'A' && up <= 'Z') {
        slots[slot++] = Tile::from_char(up);
      }
    }
    return slots;
  }

  void SetRackSlotsFromToken(int player, const std::string& rack_token) {
    racks_[player] = SlotsFromRackToken(rack_token);
  }

  // Install the resume racks parsed from top-of-file "#Rack1"/"#Rack2" pragmata
  // onto the final position, so a fully-recorded game reopens with each player's
  // recorded current rack rather than an unknown hand.
  void ApplyResumeRacks() {
    for (int p = 0; p < 2; ++p) {
      if (resume_racks_[p].has_value()) snapshots_.back().racks[p] = *resume_racks_[p];
    }
  }

  Rack RackFromSlots(const ParsedRackSlots& slots) const {
    Rack rack;
    for (int i = 0; i < RACK_SIZE; ++i) {
      if (slots[i].has_value()) rack.add(slots[i].value());
    }
    return rack;
  }

  int BagSizeEstimate() const { return std::max(0, 100 - board_tile_count(board_) - 14); }

  ParsedGcgSnapshot CurrentSnapshot(int turn_player = 0) const {
    ParsedGcgSnapshot snapshot;
    snapshot.board = board_;
    snapshot.scores = scores_;
    snapshot.racks = racks_;
    snapshot.bag = bag_;
    snapshot.turn_player = turn_player;
    return snapshot;
  }

  void FillResult(ParsedGcgGame* out_game) {
    out_game->player_names = names_;
    out_game->turns = turns_;
    out_game->snapshots = snapshots_;
    out_game->end_adjustments = end_adjustments_;
  }

  std::array<std::string, 2> names_ = {"Player 1", "Player 2"};
  std::map<std::string, int> nick_to_player_;
  Board board_;
  std::array<int, 2> scores_ = {0, 0};
  std::array<ParsedRackSlots, 2> racks_;
  std::array<std::optional<ParsedRackSlots>, 2> resume_racks_;
  TileCounts bag_;
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
    for (int p = 0; p < 2; ++p) {
      Rack rack;
      for (int i = 0; i < RACK_SIZE; ++i) {
        if (snapshots.back().racks[p][i].has_value())
          rack.add(snapshots.back().racks[p][i].value());
      }
      storage.final_racks[p] = rack;
    }
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

std::optional<Rack> pragma_rack(const std::string& gcg_text, int player) {
  const std::string want = std::format("#rack{}", player + 1);
  std::istringstream lines(gcg_text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.size() < want.size() + 1) continue;
    std::string head = line.substr(0, want.size());
    for (char& c : head) c = char(std::tolower(uint8_t(c)));
    if (head != want || line[want.size()] != ' ') continue;
    Rack rack;
    for (size_t i = want.size() + 1; i < line.size(); ++i) {
      const char c = line[i];
      if (c == ' ' || c == '\r') continue;
      rack.add(c == '?' ? BLANK : Tile::from_char(c));
    }
    return rack;
  }
  return std::nullopt;
}

bool read_gcg_endgame(const std::string& gcg_text, ParsedGcgEndgame* out,
                      std::string* error_message) {
  ParsedGcgGame game;
  if (!read_gcg_text(gcg_text, &game, error_message)) return false;
  if (game.snapshots.empty()) {
    *error_message = "GCG contains no positions";
    return false;
  }
  const ParsedGcgSnapshot& snapshot = game.snapshots.back();
  const int mover = snapshot.turn_player;

  const std::optional<Rack> mover_rack = pragma_rack(gcg_text, mover);
  if (!mover_rack.has_value()) {
    *error_message = std::format("the mover's rack is unknown: add a #Rack{} pragma", mover + 1);
    return false;
  }
  std::optional<Rack> opp_rack = pragma_rack(gcg_text, 1 - mover);
  if (!opp_rack.has_value()) {
    try {
      opp_rack = snapshot.board.hidden_rack(*mover_rack);
    } catch (const util::Exception& e) {
      *error_message = e.what();
      return false;
    }
  }

  out->board = snapshot.board;
  out->racks[mover] = *mover_rack;
  out->racks[1 - mover] = *opp_rack;
  out->scores = snapshot.scores;
  out->mover = mover;
  out->player_names = game.player_names;
  out->turns = game.turns.size();
  return true;
}

}  // namespace scribblez
