#pragma once

// Typed reads of the fields of a JSON message from a web client. A missing or
// mistyped field reads as the fallback, so a handler validates values rather
// than JSON types.

#include "game/tile.h"

#include <boost/json.hpp>

#include <string>

namespace scribblez {

int int_field(const boost::json::object& msg, boost::json::string_view key, int fallback = -1);

// Empty if absent or not a string.
std::string str_field(const boost::json::object& msg, boost::json::string_view key);

bool bool_field(const boost::json::object& msg, boost::json::string_view key,
                bool fallback = false);

// The letter A..Z (either case) a string field starts with, or EMPTY_SQUARE if
// it is absent, empty or not a letter.
Tile letter_field(const boost::json::object& msg, boost::json::string_view key);

}  // namespace scribblez
