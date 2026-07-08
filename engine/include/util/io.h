#pragma once

#include <boost/json.hpp>

#include <cstddef>
#include <ostream>
#include <string>

namespace scribblez::util {

// Read an entire file into a string. Returns an empty string if it can't be opened.
std::string read_file(const std::string& path);

// Read exactly n bytes from socket fd into buf, looping over short reads.
// Returns false on EOF or error.
bool read_n(int fd, void* buf, size_t n);

// Write exactly n bytes from buf to socket fd, looping over short writes.
// Returns false on error.
bool write_all(int fd, const void* buf, size_t n);

// Serialize `jv` with one value per line -- each object member and array element on
// its own line, nested two spaces per level -- and no trailing newline. Unlike
// boost::json::serialize, which emits everything on a single line.
void pretty_print(std::ostream& os, const boost::json::value& jv);

}  // namespace scribblez::util
