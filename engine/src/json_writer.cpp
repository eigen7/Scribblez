#include "scribblez/json_writer.h"

#include <sstream>
#include <string>

namespace scribblez {

namespace {

std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  out += "\"";
  return out;
}

std::string rack_to_string(const Rack& r) { return r.to_string(); }

std::string letters_to_string(const std::vector<Letter>& v) {
  std::string s;
  for (Letter L : v) s.push_back(L == BLANK ? '?' : letter_to_char(L));
  return s;
}

std::string move_to_json(const Move& m, const std::string& indent) {
  std::ostringstream os;
  std::string i1 = indent + "  ";
  os << "{\n";
  os << i1 << "\"type\": ";
  switch (m.type) {
    case MoveType::PLAY:     os << quote("play"); break;
    case MoveType::EXCHANGE: os << quote("exchange"); break;
    case MoveType::PASS:     os << quote("pass"); break;
  }
  if (m.type == MoveType::PLAY) {
    os << ",\n" << i1 << "\"horizontal\": " << (m.horizontal ? "true" : "false");
    os << ",\n" << i1 << "\"start_row\": " << m.start_row;
    os << ",\n" << i1 << "\"start_col\": " << m.start_col;
    os << ",\n" << i1 << "\"main_word\": " << quote(m.main_word);
    os << ",\n" << i1 << "\"score\": " << m.score;
    os << ",\n" << i1 << "\"tiles\": [";
    for (size_t k = 0; k < m.tiles.size(); ++k) {
      const auto& t = m.tiles[k];
      if (k) os << ", ";
      os << "{\"row\": " << t.row << ", \"col\": " << t.col
         << ", \"letter\": " << quote(std::string(1, letter_to_char(t.letter)))
         << ", \"is_blank\": " << (t.is_blank ? "true" : "false") << "}";
    }
    os << "]";
  } else if (m.type == MoveType::EXCHANGE) {
    os << ",\n" << i1 << "\"exchanged\": " << quote(letters_to_string(m.exchanged));
  }
  os << "\n" << indent << "}";
  return os.str();
}

}  // namespace

std::string game_log_to_json(const GameLog& log) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"seed\": " << log.seed << ",\n";
  os << "  \"players\": [";
  for (int i = 0; i < 2; ++i) {
    if (i) os << ", ";
    os << "{\"name\": " << quote(log.player_names[i]) << "}";
  }
  os << "],\n";
  os << "  \"turns\": [\n";
  for (size_t i = 0; i < log.turns.size(); ++i) {
    const auto& t = log.turns[i];
    os << "    {\n";
    os << "      \"player\": " << t.player << ",\n";
    os << "      \"rack_before\": " << quote(rack_to_string(t.rack_before)) << ",\n";
    os << "      \"bag_size_before\": " << t.bag_size_before << ",\n";
    os << "      \"move\": " << move_to_json(t.move, "      ") << ",\n";
    os << "      \"score_delta\": " << t.score_delta << ",\n";
    os << "      \"cumulative_scores\": [" << t.cumulative_scores[0]
       << ", " << t.cumulative_scores[1] << "],\n";
    os << "      \"drawn\": " << quote(letters_to_string(t.drawn)) << "\n";
    os << "    }" << (i + 1 < log.turns.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"final_scores\": [" << log.final_scores[0] << ", " << log.final_scores[1] << "],\n";
  os << "  \"end_reason\": " << quote(log.end_reason) << "\n";
  os << "}\n";
  return os.str();
}

void write_game_log_json(const GameLog& log, std::ostream& out) {
  out << game_log_to_json(log);
}

}  // namespace scribblez
