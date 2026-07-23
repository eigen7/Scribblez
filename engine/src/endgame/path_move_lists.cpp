#include "endgame/path_move_lists.h"

#include "game/glyph.h"
#include "game/movegen.h"
#include "game/tile.h"
#include "util/math.h"

namespace scribblez {

namespace {

// Add one copy of tile `t` to the packed counts (nibble layout: see
// PackedCounts).
void packed_add(PackedCounts& p, Tile t) {
  const int i = t;
  if (i < 16)
    p.lo += 1ull << (4 * i);
  else
    p.hi += 1ull << (4 * (i - 16));
}

}  // namespace

PackedCounts pack_rack(const Rack& rack) {
  PackedCounts p;
  for (int i = 0; i < rack.size(); ++i) packed_add(p, rack.tiles()[i]);
  return p;
}

PackedCounts pack_move_used(const Move& m) {
  PackedCounts p;
  for (int i = 0; i < m.num_glyphs(); ++i) packed_add(p, m.glyph(i).rack_tile());
  return p;
}

LaneTouch move_lane_influence(const Board& board, const Move& m) {
  LaneTouch t;
  visit_placed_squares(m, [&](int r, int c) {
    t.rows |= static_cast<uint16_t>(1u << r);
    t.cols |= static_cast<uint16_t>(1u << c);
    for (const auto& [dr, dc] : util::kFourNeighborDeltas) {
      int rr = r + dr, cc = c + dc;
      while (rr >= 0 && rr < BOARD_SIZE && cc >= 0 && cc < BOARD_SIZE &&
             board.at(rr, cc).has_letter()) {
        rr += dr;
        cc += dc;
      }
      if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) continue;
      t.rows |= static_cast<uint16_t>(1u << rr);
      t.cols |= static_cast<uint16_t>(1u << cc);
    }
  });
  return t;
}

void PathMoveLists::reset(const Board* board, const Dictionary* dict, int max_ply) {
  board_ = board;
  dict_ = dict;
  if (static_cast<int>(slots_.size()) < max_ply) {
    slots_.resize(max_ply);
    masks_.resize(max_ply);
    played_.resize(max_ply);
  }
}

void PathMoveLists::set_root_list(int side, const std::vector<Move>& plays) {
  fill_slot(roots_[side], plays);
}

void PathMoveLists::fill_slot(Slot& s, const std::vector<Move>& plays) {
  s.moves.assign(plays.begin(), plays.end());
  s.used.clear();
  s.used.reserve(plays.size());
  for (const Move& m : plays) s.used.push_back(pack_move_used(m));
  int lane = 0;
  s.lane_begin[0] = 0;
  for (uint32_t i = 0; i < plays.size(); ++i) {
    while (lane < lane_of(plays[i])) s.lane_begin[++lane] = i;
  }
  while (lane < 2 * BOARD_SIZE) s.lane_begin[++lane] = static_cast<uint32_t>(plays.size());
}

void PathMoveLists::on_make(int ply, const Move& m) {
  masks_[ply] = move_lane_influence(*board_, m);
  played_[ply] = m.type() == MoveType::PLAY ? 1 : 0;
}

const std::vector<Move>& PathMoveLists::moves_at(int ply, const Rack& rack) {
  if (ply == 0) return roots_[0].moves;
  // The parent is the same side's list two plies up; at plies 1 and 2 that is
  // a root list (ply 0 serves roots_[0] directly, so no slot holds it).
  const Slot& parent = ply == 1 ? roots_[1] : ply == 2 ? roots_[0] : slots_[ply - 2];
  LaneTouch touched = masks_[ply - 1];
  bool filter = false;
  if (ply >= 2) {
    touched.rows |= masks_[ply - 2].rows;
    touched.cols |= masks_[ply - 2].cols;
    filter = played_[ply - 2] != 0;
  }
  Slot& out = slots_[ply];
  rebuild(out, parent, touched, filter, rack);
  return out.moves;
}

void PathMoveLists::rebuild(Slot& out, const Slot& parent, const LaneTouch& touched, bool filter,
                            const Rack& rack) {
  out.moves.clear();
  out.used.clear();
  const PackedCounts avail = pack_rack(rack);
  MoveGenerator gen(*board_, *dict_);
  for (int lane = 0; lane < 2 * BOARD_SIZE; ++lane) {
    out.lane_begin[lane] = static_cast<uint32_t>(out.moves.size());
    const bool vertical = lane >= BOARD_SIZE;
    const int row = vertical ? lane - BOARD_SIZE : lane;
    const bool hit = ((vertical ? touched.cols : touched.rows) >> row) & 1;
    if (hit) {
      gen.generate_lane(rack, vertical, row, out.moves);
      for (size_t i = out.used.size(); i < out.moves.size(); ++i)
        out.used.push_back(pack_move_used(out.moves[i]));
    } else if (filter) {
      for (uint32_t i = parent.lane_begin[lane]; i < parent.lane_begin[lane + 1]; ++i) {
        if (!counts_subset(parent.used[i], avail)) continue;
        out.moves.push_back(parent.moves[i]);
        out.used.push_back(parent.used[i]);
      }
    } else {
      const auto mb = parent.moves.begin();
      const auto ub = parent.used.begin();
      out.moves.insert(out.moves.end(), mb + parent.lane_begin[lane],
                       mb + parent.lane_begin[lane + 1]);
      out.used.insert(out.used.end(), ub + parent.lane_begin[lane],
                      ub + parent.lane_begin[lane + 1]);
    }
  }
  out.lane_begin[2 * BOARD_SIZE] = static_cast<uint32_t>(out.moves.size());
}

}  // namespace scribblez
