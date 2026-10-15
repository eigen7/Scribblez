#include "sim/win_table_feature.h"

#include "data/binary_log.h"
#include "encoding/position_encoder.h"
#include "lexicon/hasty_equity.h"

#include <algorithm>
#include <cmath>

namespace scribblez {

namespace {

// The mover's rack after the record's move and draw.
Rack rack_after(const TurnRecord& rec) {
  Rack rack = rec.rack_before;
  for (int i = 0; i < rec.move.num_glyphs(); ++i) rack.remove(rec.move.glyph(i).rack_tile());
  for (const Tile t : rec.drawn.tiles()) {
    if (t.is_empty()) break;
    rack.add(t);
  }
  return rack;
}

float outcome_for(const GameLog& g, int player) {
  const int diff = g.final_scores[player] - g.final_scores[1 - player];
  return diff > 0 ? 1.0f : diff == 0 ? 0.5f : 0.0f;
}

// The first fittable turn. The side to move at turn t last moved at t - 2,
// so t >= 2. `eligible_begin` is the last random-opening ply's turn, or 0
// without one; a pre-move position at that turn still has the random ply
// ahead of it.
int first_fittable_turn(const binlog::GameMetadata& gm) {
  return std::max(2, gm.eligible_begin + 1);
}

WinTableSample sample_at(const GameLog& g, int t) {
  const TurnRecord& rec = g.records[t];
  const int mover = rec.player;
  const std::array<int, 2>& scores = g.records[t - 1].cumulative_scores;
  const Rack own_leave = binlog::leave_from_replay(g, t - 2, rec.rack_before);
  const int opp_rack = rack_after(g.records[t - 1]).size();
  const int spread = scores[mover] - scores[1 - mover];
  return {win_table_feature(spread, own_leave, rec.bag_size_before + opp_rack),
          outcome_for(g, mover)};
}

}  // namespace

WinTableFeature win_table_feature(int spread, const Rack& own_leave, int tiles_unseen) {
  const double leave = HastyEquity::instance().leave_value(own_leave);
  return {int(std::lround(spread + leave)), tiles_unseen};
}

void append_win_table_samples(const char* slog_buf, std::vector<WinTableSample>& out) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(slog_buf);
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(slog_buf + sizeof(binlog::FileHeader));
  std::vector<TurnRecord> scratch;
  for (uint32_t i = 0; i < hdr->num_games; ++i) {
    const GameLog g = binlog::make_game_view(slog_buf, i, scratch, nullptr);
    binlog::complete_turn_records(g, g.num_records, scratch);
    for (int t = first_fittable_turn(metas[i]); t < g.num_records; ++t) {
      out.push_back(sample_at(g, t));
    }
  }
}

}  // namespace scribblez
