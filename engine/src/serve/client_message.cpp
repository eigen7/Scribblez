#include "serve/client_message.h"

namespace scribblez {

int int_field(const boost::json::object& msg, boost::json::string_view key, int fallback) {
  auto it = msg.find(key);
  if (it == msg.end() || !it->value().is_int64()) return fallback;
  return it->value().as_int64();
}

std::string str_field(const boost::json::object& msg, boost::json::string_view key) {
  auto it = msg.find(key);
  if (it == msg.end() || !it->value().is_string()) return "";
  return std::string(it->value().as_string().c_str());
}

Tile letter_field(const boost::json::object& msg, boost::json::string_view key) {
  const std::string s = str_field(msg, key);
  return s.empty() ? EMPTY_SQUARE : Tile::letter_from_char(s[0]);
}

bool bool_field(const boost::json::object& msg, boost::json::string_view key, bool fallback) {
  auto it = msg.find(key);
  if (it == msg.end() || !it->value().is_bool()) return fallback;
  return it->value().as_bool();
}

}  // namespace scribblez
