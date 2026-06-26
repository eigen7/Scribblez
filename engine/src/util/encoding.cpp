#include "util/encoding.h"

namespace util {

std::string base64(const uint8_t* data, size_t len) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = data[i] << 16;
    if (i + 1 < len) n |= data[i + 1] << 8;
    if (i + 2 < len) n |= data[i + 2];
    out.push_back(tbl[(n >> 18) & 63]);
    out.push_back(tbl[(n >> 12) & 63]);
    out.push_back(i + 1 < len ? tbl[(n >> 6) & 63] : '=');
    out.push_back(i + 2 < len ? tbl[n & 63] : '=');
  }
  return out;
}

std::string base64_decode(const std::string& in) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int rev[256];
  for (int i = 0; i < 256; ++i) rev[i] = -1;
  for (int i = 0; i < 64; ++i) rev[static_cast<unsigned char>(tbl[i])] = i;

  std::string out;
  int val = 0;
  int bits = -8;
  for (unsigned char c : in) {
    if (c == '=') break;
    const int d = rev[c];
    if (d < 0) continue;  // skip whitespace / non-alphabet
    val = (val << 6) | d;
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((val >> bits) & 0xff));
      bits -= 8;
    }
  }
  return out;
}

}  // namespace util
