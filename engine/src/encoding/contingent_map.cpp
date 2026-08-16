#include "encoding/contingent_map.h"

#include "game/glyph.h"
#include "game/move.h"
#include "game/movegen.h"
#include "game/tile.h"

#include <algorithm>
#include <vector>

namespace scribblez {

namespace {

// Flat lane id for an assignment: row lanes 0..14, column lanes 15..29.
int lane_id(const LaneAssignment& a) { return (a.horizontal ? 0 : kLanesPerAxis) + a.lane_index; }

// The play's newly placed tiles, as a mask over the assignment's lane cells.
uint16_t lane_placed_mask(const LaneAssignment& a, const PlacedTile* placed, int num_placed) {
  uint16_t mask = 0;
  for (int k = 0; k < num_placed; ++k) {
    mask |= uint16_t(1u << (a.horizontal ? placed[k].c : placed[k].r));
  }
  return mask;
}

// Strict-max reducer for one entry slot.
void fold_entry(ContingentMap::Entry& e, int score, uint16_t mask) {
  if (score <= e.score) return;
  e.score = int16_t(score);
  e.placed_mask = mask;
}

// How many `letter` tiles the play places as natural tiles, designated blanks
// excluded.
int natural_count(const Move& m, int letter_index) {
  int n = 0;
  for (int i = 0; i < m.num_glyphs(); ++i) {
    const Glyph g = m.glyph(i);
    if (!g.is_blank() && g.letter().index() == letter_index) ++n;
  }
  return n;
}

// The extra points the play would score if the blank placed at `q` were the
// real tile of its designated letter `letter`: the letter's face value through
// the cell's letter multiplier, once per word the placement forms. The main
// word's word-multiplier product spans every newly placed cell
// (`main_word_mult`); a crossword at `q` is new only at `q`, so its product is
// the cell's own word multiplier.
int real_tile_delta(const Board& board, const Move& m, const PlacedTile& q, int num_placed,
                    int main_word_mult, Tile letter) {
  const Premium prem = board.premium_at(q.r, q.c);
  int word_factor = 0;
  const bool main_word_exists =
    num_placed >= 2 || forms_word_along_axis(board, q.r, q.c, m.horizontal());
  if (main_word_exists) word_factor += main_word_mult;
  if (forms_word_along_axis(board, q.r, q.c, !m.horizontal())) word_factor += prem.word_mult();
  return letter.value() * prem.letter_mult() * word_factor;
}

// `x` clipped to `scale` and normalized by it, giving a value in [0, 1].
float scaled_unit(float x, int scale) { return std::min(1.0f, x / float(scale)); }

// Per-cell max-paint of `value` onto the entry's placed cells within `lane`.
void paint_entry(const ContingentMap::Entry& e, int lane, float value, bool flip, float* plane) {
  if (e.score < 0) return;
  const bool horizontal = lane < kLanesPerAxis;
  const int idx = horizontal ? lane : lane - kLanesPerAxis;
  uint16_t mask = e.placed_mask;
  for (int pos = 0; mask; ++pos, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    int r = horizontal ? idx : pos;
    int c = horizontal ? pos : idx;
    if (flip) std::swap(r, c);
    float& cell = plane[r * BOARD_SIZE + c];
    cell = std::max(cell, value);
  }
}

}  // namespace

ContingentMap ContingentMap::compute(const Board& board, const Rack& rack, const uint8_t unseen[27],
                                     const Dictionary& dict) {
  ContingentMap map;
  for (int i = 0; i < kLaneTileKinds; ++i) {
    map.unseen_[i] = unseen[i];
    map.unseen_total_ += unseen[i];
  }

  // A full rack draws no replenishment tile, so no phantom is added and the
  // contingent columns stay empty -- only the rack-alone lanes populate. (This
  // also keeps the generation rack within Rack's RACK_SIZE capacity.)
  const bool has_phantom = rack.size() < RACK_SIZE;
  Rack augmented = rack;
  if (has_phantom) augmented.add(BLANK);
  MoveGenerator gen(board, dict);
  const std::vector<Move> moves = gen.generate(augmented);

  const int rack_blanks = rack.blanks();
  PlacedTile placed[RACK_SIZE];
  for (const Move& m : moves) {
    const int p = decode_placements(m, placed);
    const LaneAssignments la = compute_lane_assignments(board, m, placed, p);

    int blanks_used = 0;
    int main_word_mult = 1;
    for (int k = 0; k < p; ++k) {
      if (m.glyph(k).is_blank()) ++blanks_used;
      main_word_mult *= board.premium_at(placed[k].r, placed[k].c).word_mult();
    }

    for (int i = 0; i < la.count; ++i) {
      const LaneAssignment& a = la.items[i];
      const int lane = lane_id(a);
      const uint16_t mask = lane_placed_mask(a, placed, p);

      if (blanks_used <= rack_blanks) {
        // Every consumed blank can be the rack's own: achievable without a
        // draw.
        fold_entry(map.rack_best_[lane], m.score(), mask);
      } else if (map.unseen_[kLaneBlankKind] > 0) {
        // The play consumes the phantom as a blank: it needs a drawn blank.
        fold_entry(map.best_[kLaneBlankKind][lane], m.score(), mask);
      }
      // Independently of that split, ANY placed blank can be read as the real
      // drawn tile of its designated letter (the remaining blanks are the
      // rack's own -- with the phantom present there is always capacity).
      if (!has_phantom || blanks_used == 0) continue;
      for (int k = 0; k < p; ++k) {
        if (!m.glyph(k).is_blank()) continue;
        const Tile letter = m.glyph(k).letter();
        if (map.unseen_[letter.index()] == 0) continue;
        // Reading this blank as the real drawn letter places
        // natural_count + 1 copies; that needs the draw only when it exceeds
        // the rack's own supply -- otherwise the placement is a rack-alone
        // play (with the rack's copy here instead of a blank), not a
        // contingency.
        if (natural_count(m, letter.index()) < rack.count(letter)) continue;
        const int delta = real_tile_delta(board, m, placed[k], p, main_word_mult, letter);
        fold_entry(map.best_[letter.index()][lane], m.score() + delta, mask);
      }
    }
  }
  return map;
}

void ContingentMap::encode_planes(bool flip, float* planes_out) const {
  float* max_plane = planes_out;
  float* weighted_plane = planes_out + BOARD_SIZE * BOARD_SIZE;
  float* rack_plane = planes_out + 2 * BOARD_SIZE * BOARD_SIZE;
  for (int kind = 0; kind < kLaneTileKinds; ++kind) {
    const float p_draw = unseen_total_ > 0 ? float(unseen_[kind]) / unseen_total_ : 0.0f;
    for (int lane = 0; lane < kNumLanes; ++lane) {
      const Entry& e = best_[kind][lane];
      if (e.score < 0) continue;
      paint_entry(e, lane, scaled_unit(e.score, kContingentScoreClip), flip, max_plane);
      paint_entry(e, lane, scaled_unit(p_draw * e.score, kContingentWeightedScale), flip,
                  weighted_plane);
    }
  }
  for (int lane = 0; lane < kNumLanes; ++lane) {
    const Entry& e = rack_best_[lane];
    if (e.score < 0) continue;
    paint_entry(e, lane, scaled_unit(e.score, kContingentScoreClip), flip, rack_plane);
  }
}

void ContingentMap::encode_scalars(float* out) const {
  float expected = 0.0f;
  for (int kind = 0; kind < kLaneTileKinds; ++kind) {
    int best = -1;
    for (int lane = 0; lane < kNumLanes; ++lane)
      best = std::max(best, int{best_[kind][lane].score});
    const float p_draw = unseen_total_ > 0 ? float(unseen_[kind]) / unseen_total_ : 0.0f;
    out[kind] = best >= 0 ? scaled_unit(best, kContingentScoreClip) : 0.0f;
    out[kLaneTileKinds + kind] =
      best >= 0 ? scaled_unit(p_draw * best, kContingentWeightedScale) : 0.0f;
    if (best >= 0) expected += p_draw * best;
  }
  int rack_best = -1;
  for (int lane = 0; lane < kNumLanes; ++lane)
    rack_best = std::max(rack_best, int{rack_best_[lane].score});
  out[2 * kLaneTileKinds] = scaled_unit(expected, kContingentScoreClip);
  out[2 * kLaneTileKinds + 1] =
    rack_best >= 0 ? scaled_unit(rack_best, kContingentScoreClip) : 0.0f;
}

}  // namespace scribblez
