#pragma once

#include <stdexcept>
#include <string>

namespace scribblez {

// A user-facing error whose complete message has already been printed to
// stderr at the source, which is usually better placed to write a helpful one
// than main() is. main() catches this, exits non-zero, and prints nothing --
// while a plain std::exception still gets its "Unexpected error: " prefix.
class Exception : public std::runtime_error {
 public:
  explicit Exception(const std::string& what_arg = "") : std::runtime_error(what_arg) {}
};

// Throw to request a clean early exit with status 0 (e.g. --help was given,
// the help text was already printed, and there's nothing left to do).
class CleanExit : public Exception {
 public:
  CleanExit() = default;
};

}  // namespace scribblez
