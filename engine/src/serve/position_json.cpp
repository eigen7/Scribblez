#include "serve/position_json.h"

#include "game/glyph.h"
#include "game/tile.h"

#include <algorithm>
#include <numeric>

namespace scribblez {

namespace json = boost::json;

namespace {

// 15x15 grid: uppercase letters, lowercase for blanks, null for empty squares.
json::array board_grid(const Board& board) {
  json::array grid;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph sq = board.at(r, c);
      if (sq.is_empty()) {
        row.emplace_back(nullptr);
      } else {
        char ch = sq.letter().to_char();
        if (sq.is_blank()) ch = ch - 'A' + 'a';
        row.emplace_back(std::string(1, ch));
      }
    }
    grid.emplace_back(std::move(row));
  }
  return grid;
}

// 15x15 grid of premium codes (DL/TL/DW/TW) or null.
json::array bonus_grid(const Board& board) {
  json::array grid;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    json::array row;
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const char* code = board.premium_at(r, c).code();
      row.emplace_back(code ? json::value(code) : json::value(nullptr));
    }
    grid.emplace_back(std::move(row));
  }
  return grid;
}

// Rack as {letter, score} entries: letters first, then blanks ('?', score 0).
json::array rack_tiles(const Rack& my_rack) {
  json::array rack;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    for (int i = 0; i < my_rack.count(L); ++i) {
      rack.emplace_back(
        json::object{{"letter", std::string(1, L.to_char())}, {"score", TILE_VALUES[L]}});
    }
  }
  for (int i = 0; i < my_rack.blanks(); ++i) {
    rack.emplace_back(json::object{{"letter", "?"}, {"score", 0}});
  }
  return rack;
}

json::object tile_score_map() {
  json::object tile_scores;
  for (Tile L = Tile::of(0); L < 26; ++L) {
    tile_scores[std::string(1, L.to_char())] = TILE_VALUES[L];
  }
  return tile_scores;
}

int tiles_on_board(const Board& board) {
  int n = 0;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      if (!board.at(r, c).is_empty()) ++n;
    }
  }
  return n;
}

}  // namespace

json::array move_squares(const Move& m) {
  json::array squares;
  if (m.type() != MoveType::PLAY) return squares;
  const bool horizontal = m.horizontal();
  const int start = m.start();
  uint16_t mask = m.square_mask();
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horizontal ? start : along;
    const int c = horizontal ? along : start;
    squares.emplace_back(json::array{r, c});
  }
  return squares;
}

json::object position_state_object(const Board& board, const Rack& my_rack, int my_score,
                                   int opp_score, int bag_size, int opp_rack_size,
                                   const std::string& my_name, const std::string& opp_name,
                                   bool your_turn, bool game_over) {
  json::object o;
  o["type"] = game_over ? "game_over" : "state";
  o["board"] = board_grid(board);
  o["bonuses"] = bonus_grid(board);
  o["rack"] = rack_tiles(my_rack);
  o["scores"] = {my_score, opp_score};
  o["player_names"] = {my_name, opp_name};
  o["bag_count"] = bag_size;
  o["opponent_rack_count"] = opp_rack_size;
  o["your_turn"] = your_turn;
  o["game_over"] = game_over;
  o["tile_scores"] = tile_score_map();
  return o;
}

json::object position_state_object_pov(const Board& board, const Rack& my_rack, int my_score,
                                       int opp_score, const std::string& my_name,
                                       const std::string& opp_name) {
  // The active player cannot tell the bag from the opponent's rack; the
  // refill-to-7 rule pins the partition of the unseen pool.
  const int total = std::accumulate(TILE_COUNTS.begin(), TILE_COUNTS.end(), 0);
  const int unseen = total - tiles_on_board(board) - my_rack.size();
  const int opp_rack_size = std::min(unseen, RACK_SIZE);
  const int bag_size = unseen - opp_rack_size;
  return position_state_object(board, my_rack, my_score, opp_score, bag_size, opp_rack_size,
                               my_name, opp_name, /*your_turn=*/true, /*game_over=*/false);
}

std::string position_state_json(const Board& board, const Rack& my_rack, int my_score,
                                int opp_score, const std::string& my_name,
                                const std::string& opp_name) {
  return json::serialize(
    position_state_object_pov(board, my_rack, my_score, opp_score, my_name, opp_name));
}

}  // namespace scribblez
