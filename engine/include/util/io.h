#pragma once

#include <boost/json.hpp>

#include <cstddef>
#include <ostream>
#include <string>

namespace scribblez::util {

// Returns an empty string if the file can't be opened.
std::string read_file(const std::string& path);

// Receive/send exactly n bytes on a socket (recv/send, so not usable on
// plain files), looping over short transfers. Returns false on EOF or error.
// Sends use MSG_NOSIGNAL, so a closed peer yields false rather than SIGPIPE.
bool read_n(int fd, void* buf, size_t n);
bool write_all(int fd, const void* buf, size_t n);

// Serialize `jv` one value per line, nested two spaces per level, with no
// trailing newline -- unlike boost::json::serialize, which emits one line.
void pretty_print(std::ostream& os, const boost::json::value& jv);

}  // namespace scribblez::util
