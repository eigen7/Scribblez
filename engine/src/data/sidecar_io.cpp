#include "data/sidecar_io.h"

#include "util/exception.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <unistd.h>

namespace scribblez {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

void write_atomically(const std::string& path, const std::vector<char>& bytes) {
  const std::string tmp = std::format("{}.tmp.{}", path, ::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw util::Exception("cannot open {}", tmp);
    f.write(bytes.data(), std::streamsize(bytes.size()));
  }
  std::filesystem::rename(tmp, path);
}

const char* take_bytes(const std::vector<char>& buf, size_t* off, size_t bytes,
                       const std::string& context) {
  if (*off + bytes > buf.size()) throw util::Exception("{}: truncated", context);
  const char* at = buf.data() + *off;
  *off += bytes;
  return at;
}

}  // namespace scribblez
