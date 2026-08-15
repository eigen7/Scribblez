#include "training/lane_analysis.h"

#include "data/gcg_reader.h"
#include "game/tile.h"
#include "serve/position_json.h"
#include "training/lane_targets.h"

#include <boost/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <optional>
#include <sstream>

namespace scribblez {

namespace {

namespace json = boost::json;

// Cap on maximal plays reported per lane: a low-scoring lane can tie across many
// short plays, and the UI only needs a representative sample.
constexpr int kMaxBestMovesPerLane = 16;

// The display character for a lane-union tile kind (a letter, or '?' for a blank).
char kind_char(int kind) { return kind == kLaneBlankKind ? '?' : static_cast<char>('A' + kind); }

// One lane's placed-tile union as a 15-cell array; cell `i` lists the letter(s) /
// blank that a maximal play newly places there (empty where nothing is placed).
json::array lane_placed(const LaneBest& lane) {
  json::array cells;
  for (int cell = 0; cell < kLaneLen; ++cell) {
    json::array kinds;
    const uint32_t bits = lane.has_move ? lane.placed[cell] : 0u;
    for (int k = 0; k < kLaneTileKinds; ++k)
      if ((bits >> k) & 1u) kinds.emplace_back(std::string(1, kind_char(k)));
    cells.emplace_back(std::move(kinds));
  }
  return cells;
}

// One lane's maximal plays (capped), each with its word and origin recovered from
// the pre-move board.
json::array lane_best_moves(const Board& board, const LaneBestMoves& lane) {
  json::array out;
  const int n = std::min(static_cast<int>(lane.moves.size()), kMaxBestMovesPerLane);
  for (int i = 0; i < n; ++i) {
    const Move& m = lane.moves[i];
    const auto [row, col] = m.word_origin(board);
    out.emplace_back(json::object{
      {"word", m.main_word(board)},
      {"row", row},
      {"col", col},
      {"horizontal", m.horizontal()},
      {"score", static_cast<int>(m.score())},
    });
  }
  return out;
}

json::object lane_object(const Board& board, const LaneBest& tgt, const LaneBestMoves& bm) {
  return json::object{
    {"has_move", tgt.has_move},
    {"max_score", tgt.has_move ? tgt.max_score : 0},
    {"placed", lane_placed(tgt)},
    {"best_moves", lane_best_moves(board, bm)},
    {"num_best", static_cast<int>(bm.moves.size())},
  };
}

json::array lane_axis(const Board& board, const std::array<LaneBest, kLanesPerAxis>& tgts,
                      const std::array<LaneBestMoves, kLanesPerAxis>& bms) {
  json::array out;
  for (int i = 0; i < kLanesPerAxis; ++i) out.emplace_back(lane_object(board, tgts[i], bms[i]));
  return out;
}

// The token after a `#Rack1` / `#Rack2` header line, if present (player 0 -> Rack1).
std::optional<std::string> rack_header_token(const std::string& gcg_text, int player) {
  const std::string key = player == 0 ? "#Rack1" : "#Rack2";
  std::istringstream in(gcg_text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) != 0) continue;
    std::istringstream ls(line);
    std::string head, tok;
    ls >> head >> tok;
    return tok;  // empty string if the header has no token
  }
  return std::nullopt;
}

// A Rack from a GCG rack token: letters become tiles, '?' a blank; '_' marks an
// unknown slot (used for a hidden opponent rack) and is skipped.
Rack rack_from_header_token(const std::string& tok) {
  Rack rack;
  for (char ch : tok) {
    if (ch == '?') {
      rack.add(BLANK);
      continue;
    }
    const char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (up >= 'A' && up <= 'Z') rack.add(Tile::from_char(up));
  }
  return rack;
}

}  // namespace

bool parse_gcg_analysis_position(const std::string& gcg_text, GcgAnalysisPosition* out,
                                 std::string* error) {
  ParsedGcgGame game;
  if (!read_gcg_text(gcg_text, &game, error)) return false;
  if (game.snapshots.empty()) {
    if (error) *error = "GCG produced no position snapshots";
    return false;
  }
  const ParsedGcgSnapshot& final_pos = game.snapshots.back();
  out->board = final_pos.board;
  out->on_move = final_pos.turn_player;

  const std::optional<std::string> tok = rack_header_token(gcg_text, out->on_move);
  if (!tok) {
    if (error) *error = "missing #Rack header for the on-move player";
    return false;
  }
  out->rack = rack_from_header_token(*tok);
  return true;
}

std::string lane_analysis_json(const Board& board, const Rack& rack, int on_move,
                               const Dictionary& dict) {
  const LaneTargets targets = compute_lane_targets(board, rack, dict);
  const LaneBestMovesSet best = compute_lane_best_moves(board, rack, dict);

  const std::string my_name = std::format("Player {}", on_move + 1);
  const std::string opp_name = std::format("Player {}", 2 - on_move);
  json::object o =
    position_state_object_pov(board, rack, /*my_score=*/0, /*opp_score=*/0, my_name, opp_name);
  o["on_move"] = on_move;
  o["lane_analysis"] = json::object{
    {"rows", lane_axis(board, targets.rows, best.rows)},
    {"cols", lane_axis(board, targets.cols, best.cols)},
  };
  return json::serialize(o);
}

}  // namespace scribblez
