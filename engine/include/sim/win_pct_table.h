#pragma once

#include <array>
#include <istream>
#include <map>
#include <string>
#include <vector>

namespace scribblez {

// The format version of a scribblez win table (docs/plans/deep_bot_v0.md,
// "Versions"). Bump it whenever the grid's shape or the meaning of a cell
// (win_table_feature.h) changes; load_versioned accepts only this version.
inline constexpr int kWinTableFormatVersion = 1;

// The first line of a scribblez win table.
inline constexpr char kWinTableMagicLine[] = "# scribblez-win-table";

// An empirical win-probability table: the chance that the side to move wins,
// given the spread from its point of view and the number of tiles it cannot
// see (bag plus opponent's rack). Macondo's simmer turns a truncated
// rollout's final position into a win probability with Macondo's table
// (winpct.csv); DeepBot's sims use scribblez tables, fitted from self-play
// (py/scribblez/leave_inference/win_table.py).
//
// Both files hold the same grid: a header row, then one row per spread from
// +kMaxSpread down to -kMaxSpread, each a spread followed by a probability per
// unseen count. A scribblez table is preceded by `#` lines: the magic line,
// then `key=value` lines, among them format_version and the provenance.
class WinPctTable {
 public:
  // Spreads beyond +-kMaxSpread read as the extreme row.
  static constexpr int kMaxSpread = 300;
  // Columns for 0..kMaxUnseen unseen tiles; more read as kMaxUnseen.
  static constexpr int kMaxUnseen = 93;

  // Parses a scribblez table. Throws util::Exception on a malformed file, a
  // missing magic line, or a format_version other than
  // kWinTableFormatVersion.
  static WinPctTable load_versioned(const std::string& path);

  // Parses Macondo's unversioned winpct.csv. Throws util::Exception on a
  // malformed file.
  static WinPctTable load_macondo(const std::string& path);

  // The table bundled with the Macondo checkout, loaded on first use.
  static const WinPctTable& macondo_default();

  // P(win) for the side to move; both arguments are clamped into the table.
  float win_prob(int spread, int tiles_unseen) const;

  // A scribblez table's `key=value` header lines (format_version and
  // provenance); empty for Macondo's.
  const std::map<std::string, std::string>& header() const { return header_; }

 private:
  static constexpr int kRows = 2 * kMaxSpread + 1;
  static constexpr int kCols = kMaxUnseen + 1;

  // Parses the grid from `in`'s next line on. `path` is for errors only.
  static WinPctTable parse_grid(std::istream& in, const std::string& path);

  // Row r holds spread kMaxSpread - r.
  std::vector<std::array<float, kCols>> rows_;
  std::map<std::string, std::string> header_;
};

}  // namespace scribblez
