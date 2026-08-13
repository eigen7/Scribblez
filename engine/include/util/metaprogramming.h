#pragma once

// Consteval conveniences over P2996 reflection, factored out of consumers so
// the std::meta plumbing (access contexts, dealiasing, template-argument
// surgery, static-string promotion) reads as one call at the use site.
// Everything here is policy-free; what a consumer does with the members it
// enumerates stays with the consumer. Requires -freflection (set globally in
// the root CMakeLists).

#include <cstddef>
#include <meta>
#include <string>
#include <string_view>

namespace scribblez::util {

// All nonstatic data members of T in declaration order, private members
// included. The returned vector cannot leave constant evaluation -- consume
// it inside the same consteval function that calls this.
template <typename T>
consteval auto nonstatic_data_members() {
  return std::meta::nonstatic_data_members_of(^^T, std::meta::access_context::unchecked());
}

template <typename T>
consteval std::size_t num_members() {
  return nonstatic_data_members<T>().size();
}

// Whether `type` is T, looking through aliases on both sides.
template <typename T>
consteval bool type_is(std::meta::info type) {
  return std::meta::dealias(type) == std::meta::dealias(^^T);
}

// Whether `type` is a specialization of the class template `tmpl`
// (e.g. is_specialization_of(t, ^^std::array)).
consteval bool is_specialization_of(std::meta::info type, std::meta::info tmpl) {
  return std::meta::has_template_arguments(type) && std::meta::template_of(type) == tmpl;
}

// The element type / extent of a std::array type.
consteval std::meta::info std_array_element(std::meta::info type) {
  return std::meta::template_arguments_of(type)[0];
}
consteval std::size_t std_array_extent(std::meta::info type) {
  return std::meta::extract<std::size_t>(std::meta::template_arguments_of(type)[1]);
}

// Consteval stand-in for std::to_string, which libstdc++ has not made
// constexpr yet.
consteval std::string dec_string(std::size_t v) {
  std::string s;
  do {
    s.insert(s.begin(), static_cast<char>('0' + v % 10));
    v /= 10;
  } while (v != 0);
  return s;
}

// A string_view promoted to static storage as a NUL-terminated string, so a
// name computed during constant evaluation survives into runtime data.
consteval const char* static_string(std::string_view s) { return std::define_static_string(s); }

// A member's identifier as a reader-facing name: the trailing underscore of
// the private-member convention stripped, promoted to static storage.
consteval const char* member_name(std::meta::info member) {
  std::string n(std::meta::identifier_of(member));
  if (!n.empty() && n.back() == '_') n.pop_back();
  return std::define_static_string(n);
}

}  // namespace scribblez::util
