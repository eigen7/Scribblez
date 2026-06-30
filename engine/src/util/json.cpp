#include "util/json.h"

#include <string>

namespace util {

namespace json = boost::json;

namespace {

void print_indented(std::ostream& os, const json::value& jv, std::string& indent) {
  if (jv.kind() == json::kind::object) {
    const json::object& obj = jv.get_object();
    if (obj.empty()) {
      os << "{}";
      return;
    }
    os << "{\n";
    indent.append(2, ' ');
    for (auto it = obj.begin();;) {
      os << indent << json::serialize(it->key()) << ": ";
      print_indented(os, it->value(), indent);
      if (++it == obj.end()) break;
      os << ",\n";
    }
    indent.resize(indent.size() - 2);
    os << "\n" << indent << "}";
  } else if (jv.kind() == json::kind::array) {
    const json::array& arr = jv.get_array();
    if (arr.empty()) {
      os << "[]";
      return;
    }
    os << "[\n";
    indent.append(2, ' ');
    for (auto it = arr.begin();;) {
      os << indent;
      print_indented(os, *it, indent);
      if (++it == arr.end()) break;
      os << ",\n";
    }
    indent.resize(indent.size() - 2);
    os << "\n" << indent << "]";
  } else {
    os << json::serialize(jv);
  }
}

}  // namespace

void pretty_print(std::ostream& os, const json::value& jv) {
  std::string indent;
  print_indented(os, jv, indent);
}

}  // namespace util
