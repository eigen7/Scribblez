#include "sim/win_pct_table.h"

#include "util/exception.h"

#include <algorithm>
#include <fstream>
#include <map>
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

// Reads the `key=value` lines that follow the magic line, up to the first
// line not starting with '#'. Leaves `in` at that line.
std::map<std::string, std::string> read_header(std::istream& in, const std::string& path) {
  std::map<std::string, std::string> header;
  while (in.peek() == '#') {
    std::string line;
    std::getline(in, line);
    const size_t eq = line.find('=');
    if (line.rfind("# ", 0) != 0 || eq == std::string::npos) {
      throw util::Exception("WinPctTable: {} has a malformed header line '{}'", path, line);
    }
    header[line.substr(2, eq - 2)] = line.substr(eq + 1);
  }
  return header;
}

std::ifstream open_or_throw(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw util::Exception("WinPctTable: cannot open {}", path);
  return in;
}

}  // namespace

WinPctTable WinPctTable::parse_grid(std::istream& in, const std::string& path) {
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

WinPctTable WinPctTable::load_versioned(const std::string& path) {
  std::ifstream in = open_or_throw(path);
  std::string magic;
  std::getline(in, magic);
  if (magic != kWinTableMagicLine) {
    throw util::Exception("WinPctTable: {} is not a scribblez win table (first line '{}')", path,
                          magic);
  }
  std::map<std::string, std::string> header = read_header(in, path);
  const auto version = header.find("format_version");
  if (version == header.end() || version->second != std::to_string(kWinTableFormatVersion)) {
    throw util::Exception("WinPctTable: {} has format version {}, but this build reads {}", path,
                          version == header.end() ? "(none)" : version->second,
                          kWinTableFormatVersion);
  }
  WinPctTable table = parse_grid(in, path);
  table.header_ = std::move(header);
  return table;
}

WinPctTable WinPctTable::load_macondo(const std::string& path) {
  std::ifstream in = open_or_throw(path);
  return parse_grid(in, path);
}

const WinPctTable& WinPctTable::macondo_default() {
  static const WinPctTable table = load_macondo(kMacondoWinPct);
  return table;
}

float WinPctTable::win_prob(int spread, int tiles_unseen) const {
  const int s = std::clamp(spread, -kMaxSpread, kMaxSpread);
  const int u = std::clamp(tiles_unseen, 0, kMaxUnseen);
  return rows_[kMaxSpread - s][u];
}

}  // namespace scribblez
