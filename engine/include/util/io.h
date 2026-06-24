#pragma once

#include <cstddef>

namespace util {

// Read exactly n bytes from socket fd into buf, looping over short reads.
// Returns false on EOF or error.
bool read_n(int fd, void* buf, size_t n);

// Write exactly n bytes from buf to socket fd, looping over short writes.
// Returns false on error.
bool write_all(int fd, const void* buf, size_t n);

}  // namespace util
