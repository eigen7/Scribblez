#pragma once

#include <stdexcept>
#include <string>

namespace scribblez {

// Base class for user-facing errors that have already printed a complete,
// human-readable message to stderr at the source. The top-level main()
// catches this, returns a non-zero exit code, and prints *nothing*. Use
// this whenever the place that detects the error is in a better position
// to write a helpful message than main() is.
//
// Distinct from std::exception so a catch-all for std::exception at the top
// level can still flag genuinely unexpected failures with a "Unexpected
// error: ..." prefix.
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
