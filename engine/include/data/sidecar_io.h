#pragma once

// The byte plumbing the sidecar writers and readers share: a writer builds its
// file in memory and writes it in one go; a reader loads it whole and walks it.

#include <cstddef>
#include <string>
#include <vector>

namespace scribblez {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size);

// Writes `bytes` to `path` through a temp file and a rename, so a partial file
// never exists there: a resumed run skips the sidecars that exist, and would
// keep a truncated one.
void write_atomically(const std::string& path, const std::vector<char>& bytes);

// The next `bytes` of `buf` at `*off`, advancing it. Throws util::Exception
// naming `context` (e.g. "ProbeReader: x.sprobe") if `buf` ends first.
const char* take_bytes(const std::vector<char>& buf, size_t* off, size_t bytes,
                       const std::string& context);

}  // namespace scribblez
