#include "lexicon/hasty_equity.h"

namespace scribblez {

inline TurnLeaves::TurnLeaves(const Rack& rack, const LeaveValues& lv)
    : lv_(lv), size_(rack.size()) {
  const auto& t = rack.tiles();
  for (int i = 0; i < size_; ++i) tile_of_bit_[i] = t[i];
  TileCounts counts = rack.counts();
  int b = 0;
  for (Tile L = Tile::of(0); L <= BLANK; ++L) {
    int c = counts.count(L);
    indices_[L] = uint8_t(((1u << c) - 1u) << b);
    b += c;
  }
  full_ = uint8_t((1u << size_) - 1);
}

inline uint8_t TurnLeaves::mask_for(const Move& move) const {
  uint8_t mask = full_;
  std::array<uint8_t, 27> indices = indices_;
  for (int i = 0; i < move.num_glyphs(); ++i) {
    uint8_t& idx = indices[move.glyph(i).rack_tile().index()];
    int bit = std::countr_zero(idx);
    mask &= uint8_t(~(1u << bit));
    idx &= uint8_t(idx - 1);
  }
  return mask;
}

inline double TurnLeaves::value(uint8_t mask) {
  ensure(mask);
  return double(value_[mask]);
}

inline int TurnLeaves::point_value(uint8_t mask) {
  ensure(mask);
  return pv_[mask];
}

inline void TurnLeaves::ensure(uint8_t mask) {
  if (computed_[mask]) return;
  Rack leave;
  for (int i = 0; i < size_; ++i)
    if (mask & (1u << i)) leave.add(tile_of_bit_[i]);
  value_[mask] = lv_.lookup(leave);
  pv_[mask] = int16_t(leave.point_value());
  computed_[mask] = true;
}

}  // namespace scribblez
