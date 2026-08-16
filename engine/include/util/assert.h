#pragma once

#include "util/exception.h"

#include <source_location>

// Invariant checks that throw util::AssertionError (carrying file:line and the
// failed condition text) instead of abort()ing like assert():
//
// - RELEASE_ASSERT(cond, ...) is checked in every build. The default for
//   cold-path invariants, where an always-on check is free and a clean throw
//   beats a core dump.
// - DEBUG_ASSERT(cond, ...) is checked only in debug builds. For invariants
//   in hot inner loops (per-tile, per-square, per-sample), or whose condition
//   is itself expensive to evaluate.
//
// Both take an optional std::format() message after the condition:
//   RELEASE_ASSERT(k >= 1, "k must be >= 1, got {}", k);
// The message arguments are evaluated only when the assertion fails, so they
// may be arbitrarily expensive.
//
// Unlike assert(), a disabled DEBUG_ASSERT still compiles its arguments (the
// discarded `if constexpr` branch is fully type-checked), so release builds
// cannot bit-rot a debug-only check and its operands never draw
// unused-variable warnings.

#define DEBUG_ASSERT(COND, ...)                                                                   \
  do {                                                                                            \
    if constexpr (scribblez::util::kDebugBuild) {                                                 \
      if (!(COND)) [[unlikely]] {                                                                 \
        scribblez::util::detail::assert_fail(                                                     \
          "DEBUG_ASSERT(" #COND ")", std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
      }                                                                                           \
    }                                                                                             \
  } while (0)

#define RELEASE_ASSERT(COND, ...)                                                                 \
  do {                                                                                            \
    if (!(COND)) [[unlikely]] {                                                                   \
      scribblez::util::detail::assert_fail(                                                       \
        "RELEASE_ASSERT(" #COND ")", std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
    }                                                                                             \
  } while (0)

namespace scribblez::util {

#ifdef NDEBUG
inline constexpr bool kDebugBuild = false;
#else
inline constexpr bool kDebugBuild = true;
#endif

// Thrown by a failed DEBUG_ASSERT/RELEASE_ASSERT. A subclass of
// util::Exception: an assertion failure is always a bug.
class AssertionError : public Exception {
 public:
  using Exception::Exception;
};

namespace detail {

template <typename... Ts>
[[noreturn]] void assert_fail(const char* check, const std::source_location& loc,
                              std::format_string<Ts...> fmt, Ts&&... ts);

[[noreturn]] void assert_fail(const char* check, const std::source_location& loc);

}  // namespace detail

}  // namespace scribblez::util

#include "inlines/util/assert.inl"
