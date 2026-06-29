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

// Which lane(s) a play contributes to under the assignment rule. `horizontal`
// selects rows vs cols; `lane_index` is the row (horizontal) or column (vertical).
struct LaneAssignment {
  bool horizontal;
  int lane_index;
};

// The 0-2 lane assignments for one play: one lane for a multi-tile play (the lane
// it lies along), and -- for a single tile -- each axis in which it forms a word.
struct LaneAssignments {
  int count = 0;
  std::array<LaneAssignment, 2> items{};
  void add(bool horizontal, int lane_index) { items[count++] = {horizontal, lane_index}; }
};

LaneAssignments lane_assignments(const Board& board, const Move& m, const Placed* placed, int p) {
  LaneAssignments out;
  if (p >= 2) {
    const bool horiz = m.horizontal();
    out.add(horiz, horiz ? placed[0].r : placed[0].c);
  } else if (p == 1) {
    const Placed& q = placed[0];
    if (forms_word(board, q.r, q.c, /*horizontal=*/true)) out.add(true, q.r);
    if (forms_word(board, q.r, q.c, /*horizontal=*/false)) out.add(false, q.c);
  }
  return out;
}

// Outcome of offering a play of `score` to a lane holding its current best.
//   admit=false -> below the lane max; ignore the play.
//   reset=true  -> a new strict max; the caller must drop the prior maximum's
//                  data (union or move list) before recording this play.
struct AdmitResult {
  bool admit;
  bool reset;
};

// Fold `score` into a lane's has_move/max_score and report how to record the play.
// Shared by the union (compute_lane_targets) and move-list (compute_lane_best_moves)
// accumulators, which differ only in what data `reset` tells them to drop.
AdmitResult admit_score(bool& has_move, int& max_score, int score) {
  if (has_move && score < max_score) return {false, false};
  const bool reset = !has_move || score > max_score;
  has_move = true;
  max_score = score;
  return {true, reset};
}

}  // namespace

LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict) {
  MoveGenerator gen(board, dict);
  const std::vector<Move> moves = gen.generate(rack);

  LaneTargets targets;
  Placed placed[RACK_SIZE];
  for (const Move& m : moves) {
    const int p = generate_placements(m, placed);
    const LaneAssignments la = lane_assignments(board, m, placed, p);
    for (int i = 0; i < la.count; ++i) {
      const LaneAssignment& a = la.items[i];
      LaneBest& lane = a.horizontal ? targets.rows[a.lane_index] : targets.cols[a.lane_index];
      const AdmitResult ar = admit_score(lane.has_move, lane.max_score, m.score());
      if (!ar.admit) continue;
      if (ar.reset) lane.placed.fill(0);
      for (int k = 0; k < p; ++k)
        lane.placed[a.horizontal ? placed[k].c : placed[k].r] |= (1u << placed[k].kind);
    }
  }
  return targets;
}

LaneBestMovesSet compute_lane_best_moves(const Board& board, const Rack& rack,
                                         const Dictionary& dict) {
  MoveGenerator gen(board, dict);
  const std::vector<Move> moves = gen.generate(rack);

  LaneBestMovesSet out;
  Placed placed[RACK_SIZE];
  for (const Move& m : moves) {
    const int p = generate_placements(m, placed);
    const LaneAssignments la = lane_assignments(board, m, placed, p);
    for (int i = 0; i < la.count; ++i) {
      const LaneAssignment& a = la.items[i];
      LaneBestMoves& lane = a.horizontal ? out.rows[a.lane_index] : out.cols[a.lane_index];
      const AdmitResult ar = admit_score(lane.has_move, lane.max_score, m.score());
      if (!ar.admit) continue;
      if (ar.reset) lane.moves.clear();
      lane.moves.push_back(m);
    }
  }
  return out;
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
