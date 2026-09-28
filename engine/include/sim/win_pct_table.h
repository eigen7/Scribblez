#pragma once

#include <array>
#include <string>
#include <vector>

namespace scribblez {

// Macondo's empirical win-probability table (winpct.csv): the chance that the
// side to move wins, given the spread from its point of view and the number of
// tiles it cannot see (bag plus opponent's rack). Macondo's simmer turns a
// truncated rollout's final position into a win probability with it.
class WinPctTable {
 public:
  // Spreads beyond +-kMaxSpread read as the extreme row.
  static constexpr int kMaxSpread = 300;
  // Columns for 0..kMaxUnseen unseen tiles; more read as kMaxUnseen.
  static constexpr int kMaxUnseen = 93;

  // Parses a winpct.csv: a header row, then one row per spread from
  // +kMaxSpread down to -kMaxSpread, each a spread followed by a probability
  // per unseen count. Throws util::Exception on a malformed file.
  static WinPctTable load(const std::string& path);

  // The table bundled with the Macondo checkout, loaded on first use.
  static const WinPctTable& macondo_default();

  // P(win) for the side to move; both arguments are clamped into the table.
  float win_prob(int spread, int tiles_unseen) const;

 private:
  static constexpr int kRows = 2 * kMaxSpread + 1;
  static constexpr int kCols = kMaxUnseen + 1;

  // Row r holds spread kMaxSpread - r.
  std::vector<std::array<float, kCols>> rows_;
};

}  // namespace scribblez
