#include "training/lane_targets.h"

#include "game/glyph.h"
#include "game/move.h"
#include "game/movegen.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace scribblez {

namespace {

// The lane-union kind for a played glyph: a designated blank collapses to the
// single blank kind, every other tile maps to its letter index.
int tile_kind(Glyph g) { return g.is_blank() ? kLaneBlankKind : g.letter().index(); }

bool occupied(const Board& board, int r, int c) {
  return board.in_bounds(r, c) && !board.at(r, c).is_empty();
}

// Outcome of offering a play of `score` to a lane holding its current best.
//   admit=false -> below the lane max; ignore the play.
//   reset=true  -> a new strict max; the caller must drop the prior maximum's
//                  data (union or move list) before recording this play.
struct AdmitResult {
  bool admit;
  bool reset;
};

// Shared by the union and move-list accumulators, which differ only in what
// data `reset` tells them to drop.
AdmitResult admit_score(bool& has_move, int& max_score, int score) {
  if (has_move && score < max_score) return {false, false};
  const bool reset = !has_move || score > max_score;
  has_move = true;
  max_score = score;
  return {true, reset};
}

}  // namespace

// Decode a PLAY into its newly placed tiles. The stored glyphs are in
// ascending lane-cell order, matching square_mask().
int decode_placements(const Move& m, PlacedTile* out) {
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

bool forms_word_along_axis(const Board& board, int r, int c, bool horizontal) {
  const int dr = horizontal ? 0 : 1;
  const int dc = horizontal ? 1 : 0;
  return occupied(board, r - dr, c - dc) || occupied(board, r + dr, c + dc);
}

LaneAssignments compute_lane_assignments(const Board& board, const Move& m,
                                         const PlacedTile* placed, int num_placed) {
  LaneAssignments out;
  if (num_placed >= 2) {
    const bool horiz = m.horizontal();
    out.items[out.count++] = {horiz, horiz ? placed[0].r : placed[0].c};
  } else if (num_placed == 1) {
    const PlacedTile& q = placed[0];
    if (forms_word_along_axis(board, q.r, q.c, /*horizontal=*/true)) {
      out.items[out.count++] = {true, q.r};
    }
    if (forms_word_along_axis(board, q.r, q.c, /*horizontal=*/false)) {
      out.items[out.count++] = {false, q.c};
    }
  }
  return out;
}

LaneTargets compute_lane_targets(const Board& board, const Rack& rack, const Dictionary& dict) {
  MoveGenerator gen(board, dict);
  const std::vector<Move> moves = gen.generate(rack);

  LaneTargets targets;
  PlacedTile placed[RACK_SIZE];
  for (const Move& m : moves) {
    const int p = decode_placements(m, placed);
    const LaneAssignments la = compute_lane_assignments(board, m, placed, p);
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
  PlacedTile placed[RACK_SIZE];
  for (const Move& m : moves) {
    const int p = decode_placements(m, placed);
    const LaneAssignments la = compute_lane_assignments(board, m, placed, p);
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

// Write one lane's occupancy block, all zero where !has_move.
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

    score[h_id] = float(h.has_move ? std::min(h.max_score, kLaneScoreBins - 1) : 0);
    score[v_id] = float(v.has_move ? std::min(v.max_score, kLaneScoreBins - 1) : 0);
    mask[h_id] = h.has_move ? 1.0f : 0.0f;
    mask[v_id] = v.has_move ? 1.0f : 0.0f;
  }
}

}  // namespace scribblez
