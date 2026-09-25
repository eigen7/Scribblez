#include "sim/win_pct_table.h"

#include "util/exception.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace scribblez {

namespace {

constexpr char kMacondoWinPct[] = "/workspace/mount/macondo/data/strategy/default/winpct.csv";

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) fields.push_back(field);
  return fields;
}

}  // namespace

WinPctTable WinPctTable::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw util::Exception("WinPctTable: cannot open {}", path);
  WinPctTable table;
  table.rows_.resize(kRows);
  std::string line;
  std::getline(in, line);  // header: the unseen counts
  for (int r = 0; r < kRows; ++r) {
    if (!std::getline(in, line)) throw util::Exception("WinPctTable: {} ends at row {}", path, r);
    const std::vector<std::string> fields = split_csv_line(line);
    if (int(fields.size()) != kCols + 1 || std::stoi(fields[0]) != kMaxSpread - r) {
      throw util::Exception("WinPctTable: {} row {} is not spread {} with {} columns", path, r,
                            kMaxSpread - r, kCols);
    }
    for (int c = 0; c < kCols; ++c) table.rows_[r][c] = std::stof(fields[c + 1]);
  }
  return table;
}

const WinPctTable& WinPctTable::macondo_default() {
  static const WinPctTable table = load(kMacondoWinPct);
  return table;
}

float WinPctTable::win_prob(int spread, int tiles_unseen) const {
  const int s = std::clamp(spread, -kMaxSpread, kMaxSpread);
  const int u = std::clamp(tiles_unseen, 0, kMaxUnseen);
  return rows_[kMaxSpread - s][u];
}

}  // namespace scribblez
