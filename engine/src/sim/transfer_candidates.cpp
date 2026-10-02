#include "sim/transfer_candidates.h"

#include "game/board.h"
#include "game/tile_counts.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace scribblez {
namespace {

TileCounts tile_counts(const Move& m) {
  TileCounts t;
  for (int i = 0; i < m.num_glyphs(); ++i) t.add(m.glyph(i).rack_tile());
  return t;
}

std::vector<int> placed_squares(const Move& m) {
  std::vector<int> squares;
  visit_placed_squares(m, [&](int row, int col) { squares.push_back(row * BOARD_SIZE + col); });
  return squares;
}

bool share_a_square(const Move& a, const Move& b) {
  const std::vector<int> sb = placed_squares(b);
  return std::ranges::any_of(placed_squares(a),
                             [&](int s) { return std::ranges::contains(sb, s); });
}

bool same_tiles(const Move& a, const Move& b) {
  return tile_counts(a).to_string() == tile_counts(b).to_string();
}

bool is_play(const Move& m) { return m.type() == MoveType::PLAY; }

// One lane, an overlap, and one play's tiles the other's plus one.
bool same_lane_one_tile(const Move& a, const Move& b) {
  if (a.horizontal() != b.horizontal() || a.start() != b.start()) return false;
  if (std::abs(a.num_glyphs() - b.num_glyphs()) != 1 || !share_a_square(a, b)) return false;
  const bool a_smaller = a.num_glyphs() < b.num_glyphs();
  TileCounts larger = tile_counts(a_smaller ? b : a);
  return larger.remove(tile_counts(a_smaller ? a : b));
}

// A selection in progress: indices into the ranking, in the order taken.
class Picker {
 public:
  Picker(const std::vector<Move>& ranked, const TransferRecipe& recipe, std::mt19937_64& rng)
      : ranked_(ranked), recipe_(recipe), rng_(rng), taken_(ranked.size(), 0) {}

  void take_couplings();
  void fill_strata();
  void fill_rest();
  SimCandidates result() const;

 private:
  bool full() const { return int(chosen_.size()) >= recipe_.size(); }
  void take(int i);
  int quota(Stratum s) const;
  Stratum stratum(int i) const { return stratum_of(ranked_[i], i, recipe_); }
  void take_random(std::vector<int> pool, int n);

  const std::vector<Move>& ranked_;
  const TransferRecipe& recipe_;
  std::mt19937_64& rng_;
  std::vector<char> taken_;
  std::vector<int> chosen_;
};

void Picker::take(int i) {
  if (taken_[i] || full()) return;
  taken_[i] = 1;
  chosen_.push_back(i);
}

int Picker::quota(Stratum s) const {
  switch (s) {
    case Stratum::kTop:
      return recipe_.top;
    case Stratum::kMiddle:
      return recipe_.middle;
    case Stratum::kExchange:
      return recipe_.exchanges;
    case Stratum::kLow:
      return recipe_.low;
  }
  return 0;
}

void Picker::take_random(std::vector<int> pool, int n) {
  std::erase_if(pool, [&](int i) { return taken_[i]; });
  std::shuffle(pool.begin(), pool.end(), rng_);
  for (int k = 0; k < n && k < int(pool.size()); ++k) take(pool[k]);
}

// One random pair of each kind the position offers.
void Picker::take_couplings() {
  for (const auto& kind_pairs : anchored_couplings(ranked_, recipe_)) {
    if (kind_pairs.empty()) continue;
    const auto& [a, b] =
      kind_pairs[std::uniform_int_distribution<size_t>(0, kind_pairs.size() - 1)(rng_)];
    take(a);
    take(b);
  }
}

void Picker::fill_strata() {
  for (const Stratum s : {Stratum::kTop, Stratum::kMiddle, Stratum::kExchange, Stratum::kLow}) {
    std::vector<int> pool;
    int have = 0;
    for (int i = 0; i < int(ranked_.size()); ++i) {
      if (stratum(i) != s) continue;
      if (taken_[i]) ++have;
      pool.push_back(i);
    }
    take_random(std::move(pool), quota(s) - have);
  }
}

void Picker::fill_rest() {
  std::vector<int> pool(ranked_.size());
  for (int i = 0; i < int(pool.size()); ++i) pool[i] = i;
  take_random(std::move(pool), recipe_.size() - int(chosen_.size()));
}

SimCandidates Picker::result() const {
  std::vector<int> order = chosen_;
  std::ranges::sort(order);
  SimCandidates out;
  out.num_legal_moves = ranked_.size();
  for (const int i : order) {
    out.moves.push_back(ranked_[i]);
    out.equity_ranks.push_back(i);
  }
  out.highlighted.assign(out.moves.size(), 0);
  return out;
}

}  // namespace

Stratum stratum_of(const Move& m, int equity_rank, const TransferRecipe& recipe) {
  if (m.type() == MoveType::EXCHANGE) return Stratum::kExchange;
  if (equity_rank < recipe.top_ranks) return Stratum::kTop;
  if (equity_rank < recipe.middle_ranks) return Stratum::kMiddle;
  return Stratum::kLow;
}

Coupling coupling_of(const Move& a, const Move& b) {
  if (is_play(a) != is_play(b)) {
    const Move& play = is_play(a) ? a : b;
    const Move& other = is_play(a) ? b : a;
    const bool exchanged = other.type() == MoveType::EXCHANGE && play.num_glyphs() > 0;
    return exchanged && same_tiles(a, b) ? Coupling::kPlayExchange : Coupling::kNone;
  }
  if (!is_play(a)) return Coupling::kNone;
  if (same_tiles(a, b) && placed_squares(a) != placed_squares(b)) return Coupling::kSameTiles;
  if (same_lane_one_tile(a, b)) return Coupling::kSameLaneOneTile;
  return Coupling::kNone;
}

AnchoredCouplings anchored_couplings(const std::vector<Move>& ranked,
                                     const TransferRecipe& recipe) {
  const int anchors = std::min(int(ranked.size()), recipe.middle_ranks);
  AnchoredCouplings pairs;
  for (int i = 0; i < anchors; ++i) {
    if (!is_play(ranked[i])) continue;
    for (int j = 0; j < int(ranked.size()); ++j) {
      // A pair of two anchors is found once, from its lower-ranked anchor.
      if (j == i || (j < i && is_play(ranked[j]))) continue;
      const Coupling kind = coupling_of(ranked[i], ranked[j]);
      if (kind != Coupling::kNone) pairs[int(kind) - 1].push_back({i, j});
    }
  }
  return pairs;
}

std::vector<CoupledPair> find_couplings(const std::vector<Move>& moves) {
  std::vector<CoupledPair> out;
  for (int a = 0; a < int(moves.size()); ++a) {
    for (int b = a + 1; b < int(moves.size()); ++b) {
      const Coupling kind = coupling_of(moves[a], moves[b]);
      if (kind != Coupling::kNone) out.push_back({a, b, kind});
    }
  }
  return out;
}

SimCandidateSelector transfer_selector(const TransferRecipe& recipe) {
  return [recipe](const binlog::GamePositionIndex&, const SimPosition&,
                  const std::vector<Move>& ranked, const Move&, std::mt19937_64& rng) {
    Picker picker(ranked, recipe, rng);
    picker.take_couplings();
    picker.fill_strata();
    picker.fill_rest();
    return picker.result();
  };
}

}  // namespace scribblez
