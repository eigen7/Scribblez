#pragma once

#include <boost/json.hpp>

#include <ostream>

namespace util {

// Serialize `jv` with one value per line -- each object member and array element on
// its own line, nested two spaces per level -- and no trailing newline. Unlike
// boost::json::serialize, which emits everything on a single line.
void pretty_print(std::ostream& os, const boost::json::value& jv);

}  // namespace util
