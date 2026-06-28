#include "scribblez/lane_targets.h"

#include "scribblez/glyph.h"
#include "scribblez/move.h"
#include "scribblez/movegen.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace scribblez {

namespace {

// The lane-union kind for a played glyph: a designated blank collapses to the
// single blank kind, every other tile maps to its letter index.
int tile_kind(Glyph g) { return g.is_blank() ? kLaneBlankKind : g.letter().index(); }

// A single newly placed tile, in absolute board coordinates.
struct Placed {
  int r;
  int c;
  int kind;
};

bool occupied(const Board& board, int r, int c) {
  return board.in_bounds(r, c) && !board.at(r, c).is_empty();
}

// Whether placing a tile at (r, c) forms a word of length >= 2 along the given
// axis -- i.e. whether the play "scores in that direction". True iff the square
// has an occupied neighbor along that axis; for a legal play any such run is a
// valid word.
bool forms_word(const Board& board, int r, int c, bool horizontal) {
  const int dr = horizontal ? 0 : 1;
  const int dc = horizontal ? 1 : 0;
  return occupied(board, r - dr, c - dc) || occupied(board, r + dr, c + dc);
}

// Decode a PLAY into its newly placed tiles. `start()` is the lane's cross-axis
// coordinate and `square_mask()` marks the placed cells along the lane; the
// stored glyphs are in ascending lane-cell order.
int generate_placements(const Move& m, Placed* out) {
  const bool horiz = m.horizontal();
  uint16_t mask = m.square_mask();
  int gi = 0;
  for (int pos = 0; mask; ++pos, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const int r = horiz ? m.start() : pos;
    const int c = horiz ? pos : m.start();
    out[gi] = {r, c, tile_kind(m.glyph(gi))};
    ++gi;
  }
  return gi;
}

// Prepare `lane` to receive a play of `score`: returns false if the play scores
// below the lane's current best (nothing to record); on a strict improvement it
// resets the union so only the new maximum's tiles remain.
bool admit(LaneBest& lane, int score) {
  if (lane.has_move && score < lane.max_score) return false;
  if (!lane.has_move || score > lane.max_score) {
    lane.has_move = true;
    lane.max_score = score;
    lane.placed.fill(0);
  }
  return true;
}

// Fold a multi-tile play into the single lane it lies along.
void offer_multi(LaneTargets& t, const Move& m, const Placed* begin, const Placed* end) {
  const bool horiz = m.horizontal();
  LaneBest& lane = horiz ? t.rows[begin->r] : t.cols[begin->c];
  if (!admit(lane, m.score())) return;
  for (const Placed* q = begin; q != end; ++q) lane.placed[horiz ? q->c : q->r] |= (1u << q->kind);
}

// Fold a single-tile play into each lane in which it forms a word.
void offer_single(LaneTargets& t, const Board& board, const Move& m, const Placed& q) {
  if (forms_word(board, q.r, q.c, /*horizontal=*/true)) {
    LaneBest& lane = t.rows[q.r];
    if (admit(lane, m.score())) lane.placed[q.c] |= (1u << q.kind);
  }
  if (forms_word(board, q.r, q.c, /*horizontal=*/false)) {
    LaneBest& lane = t.cols[q.c];
    if (admit(lane, m.score())) lane.placed[q.r] |= (1u << q.kind);
  }
}

}  // namespace

LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict) {
  MoveGenerator gen(board, dict);
  const std::vector<Move> moves = gen.generate(rack);

  LaneTargets targets;

  Placed placed[RACK_SIZE];
  for (const Move& m : moves) {
    int p = generate_placements(m, placed);
    if (p >= 2)
      offer_multi(targets, m, placed, placed + p);
    else if (p == 1)
      offer_single(targets, board, m, placed[0]);
  }
  return targets;
}

namespace {

// Write one lane's 15x27 occupancy block (zero where !has_move).
void encode_lane_occupancy(const LaneBest& lane, float* out) {
  for (int cell = 0; cell < kLaneLen; ++cell) {
    const uint32_t bits = lane.has_move ? lane.placed[cell] : 0u;
    for (int kind = 0; kind < kLaneTileKinds; ++kind)
      out[cell * kLaneTileKinds + kind] = (bits >> kind) & 1u ? 1.0f : 0.0f;
  }
}

}  // namespace

void encode_lane_targets(const LaneTargets& t, bool flip, float* out) {
  float* occ = out;
  float* score = out + kLaneOccupancyFloats;
  float* mask = score + kLaneScoreFloats;

  // Flat lane id `axis * 15 + lane`: axis 0 = horizontal lanes, axis 1 = vertical
  // lanes. A diagonal flip swaps the two axes (see header).
  const auto& horizontal = flip ? t.cols : t.rows;
  const auto& vertical = flip ? t.rows : t.cols;

  for (int lane = 0; lane < kLanesPerAxis; ++lane) {
    const LaneBest& h = horizontal[lane];
    const LaneBest& v = vertical[lane];
    const int h_id = lane;
    const int v_id = kLanesPerAxis + lane;

    encode_lane_occupancy(h, occ + h_id * kLaneLen * kLaneTileKinds);
    encode_lane_occupancy(v, occ + v_id * kLaneLen * kLaneTileKinds);

    score[h_id] = static_cast<float>(h.has_move ? std::min(h.max_score, kLaneScoreBins - 1) : 0);
    score[v_id] = static_cast<float>(v.has_move ? std::min(v.max_score, kLaneScoreBins - 1) : 0);
    mask[h_id] = h.has_move ? 1.0f : 0.0f;
    mask[v_id] = v.has_move ? 1.0f : 0.0f;
  }
}

}  // namespace scribblez
