#include "scribblez/json_writer.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

std::string letters_to_string(const std::vector<Tile>& v) {
  std::string s;
  for (Tile L : v) s.push_back(L == BLANK ? '?' : tile_to_char(L));
  return s;
}

json::object move_to_json(const Move& m) {
  json::object o;
  switch (m.type) {
    case MoveType::PLAY:
      o["type"] = "play";
      break;
    case MoveType::EXCHANGE:
      o["type"] = "exchange";
      break;
    case MoveType::PASS:
      o["type"] = "pass";
      break;
  }
  if (m.type == MoveType::PLAY) {
    o["horizontal"] = m.horizontal;
    o["start_row"] = m.start_row;
    o["start_col"] = m.start_col;
    o["main_word"] = m.main_word;
    o["score"] = m.score;
    json::array tiles;
    for (const auto& t : m.tiles) {
      tiles.push_back(json::object{{"row", t.row},
                                   {"col", t.col},
                                   {"letter", std::string(1, tile_to_char(t.letter))},
                                   {"is_blank", t.is_blank}});
    }
    o["tiles"] = std::move(tiles);
  } else if (m.type == MoveType::EXCHANGE) {
    o["exchanged"] = letters_to_string(m.exchanged);
  }
  return o;
}

}  // namespace

std::string game_log_to_json(const GameLog& log) {
  json::object root;
  root["seed"] = log.seed;

  json::array players;
  for (int i = 0; i < 2; ++i) {
    players.push_back(json::object{{"name", log.player_names[i]}});
  }
  root["players"] = std::move(players);

  json::array turns;
  for (const auto& t : log.turns) {
    json::object turn;
    turn["player"] = t.player;
    turn["rack_before"] = t.rack_before.to_string();
    turn["bag_size_before"] = t.bag_size_before;
    turn["move"] = move_to_json(t.move);
    turn["score_delta"] = t.score_delta;
    turn["cumulative_scores"] = {t.cumulative_scores[0], t.cumulative_scores[1]};
    turn["drawn"] = letters_to_string(t.drawn);
    turns.push_back(std::move(turn));
  }
  root["turns"] = std::move(turns);

  root["final_scores"] = {log.final_scores[0], log.final_scores[1]};
  root["end_reason"] = log.end_reason;

  return json::serialize(root);
}

void write_game_log_json(const GameLog& log, std::ostream& out) { out << game_log_to_json(log); }

}  // namespace scribblez
