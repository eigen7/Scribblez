#include "util/io.h"

#include <sys/socket.h>

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>

namespace scribblez::util {

namespace json = boost::json;

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool read_n(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < n) {
    ssize_t r = ::recv(fd, p + got, n - got, 0);
    if (r <= 0) return false;
    got += static_cast<size_t>(r);
  }
  return true;
}

bool write_all(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < n) {
    ssize_t w = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
    if (w <= 0) return false;
    sent += static_cast<size_t>(w);
  }
  return true;
}

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

}  // namespace scribblez::util
