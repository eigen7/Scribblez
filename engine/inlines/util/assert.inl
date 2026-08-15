#include "util/assert.h"

namespace scribblez::util::detail {

template <typename... Ts>
[[noreturn]] void assert_fail(const char* check, const std::source_location& loc,
                              std::format_string<Ts...> fmt, Ts&&... ts) {
  throw AssertionError("{} failed at {}:{}: {}", check, loc.file_name(), loc.line(),
                       std::format(fmt, std::forward<Ts>(ts)...));
}

[[noreturn]] inline void assert_fail(const char* check, const std::source_location& loc) {
  throw AssertionError("{} failed at {}:{}", check, loc.file_name(), loc.line());
}

}  // namespace scribblez::util::detail
