#pragma once

#include <format>
#include <stdexcept>
#include <utility>

namespace scribblez::util {

// The engine's general error type: std::runtime_error mechanics, with the
// message built by std::format(). Escaping to main() uncaught, it is reported
// as an unexpected error -- a bug or a broken environment; see
// main_exit_code() in util/misc.h for the full taxonomy-to-exit-code mapping.
class Exception : public std::runtime_error {
 public:
  template <typename... Ts>
  explicit Exception(std::format_string<Ts...> fmt, Ts&&... ts)
      : std::runtime_error(std::format(fmt, std::forward<Ts>(ts)...)) {}
};

// A user-facing error that is not a bug: bad flags, an unreadable input file,
// a missing model. main() prints the message as "Error: <what>" -- without
// the alarming "Unexpected" framing -- and exits non-zero.
class CleanException : public Exception {
 public:
  using Exception::Exception;
};

// Throw to request a clean early exit with status 0 (e.g. --help was given,
// the help text was already printed, and there's nothing left to do).
class CleanExit : public CleanException {
 public:
  CleanExit() : CleanException("") {}
};

}  // namespace scribblez::util
