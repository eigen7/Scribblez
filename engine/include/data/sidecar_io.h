#pragma once

// The byte plumbing the sidecar writers share: each builds its file in memory
// and writes it in one go.

#include <cstddef>
#include <string>
#include <vector>

namespace scribblez {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size);

// Writes `bytes` to `path` through a temp file and a rename, so a partial file
// never exists there: a resumed run skips the sidecars that exist, and would
// keep a truncated one.
void write_atomically(const std::string& path, const std::vector<char>& bytes);

}  // namespace scribblez
