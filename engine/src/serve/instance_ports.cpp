#include "serve/instance_ports.h"

#include <cstdlib>
#include <string>

namespace scribblez {

int instance_port_offset() {
  const char* raw = std::getenv("DEVENV_INSTANCE_PORT_OFFSET");
  if (raw == nullptr) return 0;
  try {
    size_t consumed = 0;
    int offset = std::stoi(raw, &consumed);
    // Reject trailing garbage and negative offsets; treat as "no offset".
    if (consumed != std::string(raw).size() || offset < 0) return 0;
    return offset;
  } catch (const std::exception&) {
    return 0;
  }
}

}  // namespace scribblez
