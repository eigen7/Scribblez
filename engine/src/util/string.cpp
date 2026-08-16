#include "util/string.h"

#include <array>

namespace scribblez::util {

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

namespace {

constexpr std::array<uint32_t, 5> kSha1Init = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u,
                                               0xC3D2E1F0u};

uint32_t rotl(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }

// Append the SHA-1 padding: a 0x80 byte, zeroes up to a 56-mod-64 boundary,
// then the message length in bits as a big-endian 64-bit integer.
void sha1_pad(std::string& data, uint64_t bit_len) {
  data.push_back(char(0x80));
  while (data.size() % 64 != 56) data.push_back(0);
  for (int i = 7; i >= 0; --i) data.push_back(char((bit_len >> (i * 8)) & 0xff));
}

// Mix one 64-byte block into the running hash state `h`.
void sha1_process_chunk(std::array<uint32_t, 5>& h, const char* chunk) {
  uint32_t w[80];
  for (int i = 0; i < 16; ++i) {
    w[i] = (uint8_t(chunk[i * 4]) << 24) | (uint8_t(chunk[i * 4 + 1]) << 16) |
           (uint8_t(chunk[i * 4 + 2]) << 8) | (uint8_t(chunk[i * 4 + 3]));
  }
  for (int i = 16; i < 80; ++i) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
  for (int i = 0; i < 80; ++i) {
    uint32_t f, k;
    if (i < 20) {
      f = (b & c) | (~b & d);
      k = 0x5A827999u;
    } else if (i < 40) {
      f = b ^ c ^ d;
      k = 0x6ED9EBA1u;
    } else if (i < 60) {
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDCu;
    } else {
      f = b ^ c ^ d;
      k = 0xCA62C1D6u;
    }
    uint32_t tmp = rotl(a, 5) + f + e + k + w[i];
    e = d;
    d = c;
    c = rotl(b, 30);
    b = a;
    a = tmp;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
}

// Serialize the hash state to 20 big-endian bytes.
void sha1_finalize(const std::array<uint32_t, 5>& h, uint8_t out[20]) {
  for (int i = 0; i < 5; ++i) {
    out[i * 4] = (h[i] >> 24) & 0xff;
    out[i * 4 + 1] = (h[i] >> 16) & 0xff;
    out[i * 4 + 2] = (h[i] >> 8) & 0xff;
    out[i * 4 + 3] = h[i] & 0xff;
  }
}

}  // namespace

void sha1(const std::string& msg, uint8_t out[20]) {
  std::array<uint32_t, 5> h = kSha1Init;
  std::string data = msg;
  sha1_pad(data, uint64_t(msg.size()) * 8);
  for (size_t chunk = 0; chunk < data.size(); chunk += 64) {
    sha1_process_chunk(h, data.data() + chunk);
  }
  sha1_finalize(h, out);
}

}  // namespace scribblez::util
