#include "scribblez/rack.h"

namespace scribblez {

std::string Rack::to_string() const {
  std::string s;
  for (Letter l = 0; l < 26; ++l) {
    for (int i = 0; i < counts_[l]; ++i) s.push_back(letter_to_char(l));
  }
  for (int i = 0; i < counts_[BLANK]; ++i) s.push_back('?');
  return s;
}

int Rack::point_value() const {
  int v = 0;
  for (Letter l = 0; l < 26; ++l) v += counts_[l] * LETTER_VALUES[l];
  return v;
}

}  // namespace scribblez
