#include "data/probe_replay.h"

#include "data/binary_log.h"
#include "encoding/position_encoder.h"
#include "game/bag.h"
#include "game/rack.h"
#include "util/exception.h"

#include <algorithm>
#include <array>

namespace scribblez {
namespace {

void write_tiles(const Rack& r, char* out) {
  const std::array<Tile, RACK_SIZE>& tiles = r.tiles();
  for (int i = 0; i < RACK_SIZE; ++i) out[i] = char(tiles[size_t(i)].index());
}

// `whole` minus the tiles of `part`, which must all be on it.
Rack rack_minus(Rack whole, const Rack& part) {
  for (const Tile t : part.tiles()) {
    if (t.is_empty()) break;
    if (!whole.remove(t)) {
      throw util::Exception("probe replay: {} does not hold {}", whole.to_string(),
                            part.to_string());
    }
  }
  return whole;
}

Rack leave_after(Rack rack, const Move& m) {
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  return rack;
}

int16_t score_diff(const std::array<int, 2>& scores, int mover) {
  return int16_t(scores[size_t(mover)] - scores[size_t(1 - mover)]);
}

// The state before turn `turn` of `game`, its mover to play.
binlog::ReplayStart replay_to(const GameLog& game, int turn) {
  std::vector<TurnRecord> records(game.records, game.records + turn);
  binlog::ReplayStart start;
  start.racks = game.initial_racks;
  start.scores = game.initial_scores;
  start.bag_size = Bag::kTotalTiles - game.initial_racks[0].size() - game.initial_racks[1].size();
  return binlog::replay_turn_records(start, records.data(), turn);
}

ProbeRootState root_state(const binlog::ReplayStart& root, const Rack& opp_leave) {
  const int mover = root.first_player;
  ProbeRootState s{};
  s.score_diff = score_diff(root.scores, mover);
  s.bag_size = uint8_t(root.bag_size);
  s.mover = uint8_t(mover);
  write_tiles(root.racks[size_t(mover)], s.rack);
  write_tiles(opp_leave, s.opp_leave);
  return s;
}

// The state after `candidate`, the root mover's rack holding only the leave.
binlog::ReplayStart after_candidate(const binlog::ReplayStart& root, const Move& candidate) {
  const int mover = root.first_player;
  binlog::ReplayStart s = root;
  s.racks[size_t(mover)] = leave_after(root.racks[size_t(mover)], candidate);
  if (candidate.type() == MoveType::PLAY) {
    s.scores[size_t(mover)] += candidate.score();
    s.bag_size -= std::min(candidate.num_glyphs(), root.bag_size);
  }
  s.first_player = 1 - mover;
  return s;
}

ProbeCandidateState candidate_state(const binlog::ReplayStart& after, int mover) {
  ProbeCandidateState s{};
  write_tiles(after.racks[size_t(mover)], s.leave);
  s.score_diff = score_diff(after.scores, mover);
  s.bag_size = uint8_t(after.bag_size);
  return s;
}

ProbeStartState start_state(const Rack& leave, const Rack& opp_leave, const ProbeRecord& rec) {
  ProbeStartState s{};
  write_tiles(rack_minus(rec.mover_rack, leave), s.mover_drawn);
  write_tiles(rec.mover_rack, s.mover_rack);
  write_tiles(rack_minus(rec.opp_rack, opp_leave), s.opp_drawn);
  write_tiles(rec.opp_rack, s.opp_rack);
  return s;
}

ProbeTurnState turn_state(const TurnRecord& rec, int root_mover, int ply) {
  ProbeTurnState s{};
  const Rack leave = leave_after(rec.rack_before, rec.move);
  Rack after = leave;
  for (const Tile t : rec.drawn.tiles()) {
    if (t.is_empty()) break;
    after.add(t);
  }
  write_tiles(leave, s.leave);
  write_tiles(rec.drawn, s.drawn);
  write_tiles(after, s.rack_after);
  s.root_mover = rec.player == root_mover ? 1 : 0;
  s.ply = uint8_t(ply);
  s.bag_size = uint8_t(rec.bag_size_before);
  std::array<int, 2> before = rec.cumulative_scores;
  before[size_t(rec.player)] -= rec.score_delta;
  s.score_diff = score_diff(before, root_mover);
  return s;
}

// Appends one record's turns, replayed from `after` with the record's racks.
void replay_record(binlog::ReplayStart after, int mover, const ProbeRecord& rec,
                   const binlog::TurnBlob* blobs, ProbeReplay* out) {
  after.racks[size_t(mover)] = rec.mover_rack;
  after.racks[size_t(1 - mover)] = rec.opp_rack;
  std::vector<TurnRecord> turns(rec.num_turns);
  for (size_t k = 0; k < turns.size(); ++k) {
    turns[k].move = blobs[k].move;
    turns[k].drawn = blobs[k].drawn;
  }
  binlog::replay_turn_records(after, turns.data(), int(turns.size()));
  for (size_t k = 0; k < turns.size(); ++k)
    out->turns.push_back(turn_state(turns[k], mover, int(k) + 1));
}

}  // namespace

void replay_probe_position(const GameLog& game, const ProbeReader::Position& pos, int probes,
                           ProbeReplay* out) {
  const int turn = int(pos.header->turn_index);
  const binlog::ReplayStart root = replay_to(game, turn);
  const int mover = root.first_player;
  const Rack opp_leave = binlog::opp_leave_from_replay(game, turn, root.racks[size_t(1 - mover)]);
  out->roots.push_back(root_state(root, opp_leave));
  const binlog::TurnBlob* blobs = pos.turns;
  for (uint32_t c = 0; c < pos.header->num_candidates; ++c) {
    const binlog::ReplayStart after = after_candidate(root, pos.candidates[c].move);
    const Rack& leave = after.racks[size_t(mover)];
    out->candidates.push_back(candidate_state(after, mover));
    for (int i = 0; i < probes; ++i) {
      const ProbeRecord& rec = pos.records[c * uint32_t(probes) + uint32_t(i)];
      out->starts.push_back(start_state(leave, opp_leave, rec));
      replay_record(after, mover, rec, blobs, out);
      blobs += rec.num_turns;
    }
  }
}

ProbeReplay replay_probe_file(const char* slog, const ProbeReader& probes) {
  ProbeReplay out;
  std::vector<TurnRecord> scratch;
  for (int i = 0; i < probes.num_positions(); ++i) {
    const ProbeReader::Position pos = probes.position(i);
    const GameLog game = binlog::make_game_view(slog, pos.header->game_index, scratch, nullptr);
    replay_probe_position(game, pos, probes.probes(), &out);
  }
  return out;
}

}  // namespace scribblez
