#include "scribblez/binary_log.h"

#include "scribblez/board.h"
#include "scribblez/game.h"
#include "scribblez/tile.h"
#include "scribblez/unique_id.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>

namespace scribblez {
namespace binlog {

namespace {

// Apply a played move to a Rack copy (mirrors Game::play()'s rack mutation
// for the PLAY branch, which is the only branch that produces a post-move
// snapshot).
void remove_played_tiles_from_rack(Rack& rack, const Move& m) {
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) {
    [[maybe_unused]] bool ok = rack.remove(m.glyphs[i].rack_tile());
    assert(ok);
  }
}

void remove_exchanged_tiles_from_rack(Rack& rack, const Move& m) {
  remove_played_tiles_from_rack(rack, m);  // same operation
}

// Fill bag_counts as TILE_COUNTS minus all tiles currently on `board` and in
// either rack. Every tile is in exactly one of (bag, board, racks).
void compute_bag_counts(uint8_t out[27], const Board& board, const Rack& r0, const Rack& r1) {
  // Initialize from the static distribution.
  for (int i = 0; i < 27; ++i) out[i] = static_cast<uint8_t>(TILE_COUNTS[i]);

  // Subtract tiles on the board. A designated blank on the board consumed a
  // blank tile from a rack (Glyph::rack_tile() returns BLANK for any blank).
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      Tile t = g.rack_tile();
      assert(out[t] > 0);
      --out[t];
    }
  }

  // Subtract tiles on each rack.
  for (const Rack* rk : {&r0, &r1}) {
    for (Tile t : rk->tiles()) {
      if (t.is_empty()) continue;
      assert(out[t] > 0);
      --out[t];
    }
  }
}

void fill_board_array(Glyph dst[225], const Board& board) {
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      dst[r * BOARD_SIZE + c] = board.at(r, c);
    }
  }
}

// One eligible position during a replay.
struct EligibleRef {
  int turn_index;  // index into log.turns
  PositionKind kind;
};

// Build a PositionRecord for the given eligible reference, using the
// already-prepared replay state (board / racks / last_opp_move at the
// pre-move moment of that turn).
PositionRecord make_record(const GameLog& log, const EligibleRef& ref, const Board& board_pre,
                           const Rack& rack_active_pre, const Rack& rack_opp,
                           const Move& last_opp_move) {
  const TurnRecord& turn = log.turns[ref.turn_index];
  const int active = turn.player;
  const int opp = 1 - active;

  PositionRecord rec{};
  rec.active_player = static_cast<uint8_t>(active);
  rec.position_kind = static_cast<uint8_t>(ref.kind);
  rec.move_number = static_cast<int16_t>(ref.turn_index);

  // Cumulative scores at the pre-move moment = cumulative scores BEFORE this
  // turn's score_delta is added. For pre-move that's the score going in; for
  // post-move (PLAY only) we add the move's score to the active player.
  const int prev_active = turn.cumulative_scores[active] - turn.score_delta;
  const int prev_opp = turn.cumulative_scores[opp];

  rec.score_active = prev_active;
  rec.score_opp = prev_opp;
  rec.last_opp_move = last_opp_move;
  rec.active_rack = rack_active_pre;

  Board board_for_record = board_pre;
  Rack rack_for_record = rack_active_pre;

  if (ref.kind == PositionKind::kPostMove) {
    // Only valid for PLAY (we never enqueue post-move for exchange/pass).
    assert(turn.move.type == MoveType::PLAY);
    board_for_record.apply(turn.move);
    remove_played_tiles_from_rack(rack_for_record, turn.move);
    rec.active_rack = rack_for_record;
    rec.score_active = prev_active + turn.score_delta;
  }

  compute_bag_counts(rec.bag_counts, board_for_record, rack_for_record, rack_opp);
  fill_board_array(rec.board, board_for_record);

  return rec;
}

}  // namespace

std::vector<PositionRecord> extract_positions(const GameLog& log, int samples_per_game) {
  // ---- pass 1: enumerate eligible positions -------------------------------
  std::vector<EligibleRef> eligible;
  eligible.reserve(log.turns.size() * 2);
  for (size_t i = 0; i < log.turns.size(); ++i) {
    eligible.push_back({static_cast<int>(i), PositionKind::kPreMove});
    if (log.turns[i].move.type == MoveType::PLAY) {
      eligible.push_back({static_cast<int>(i), PositionKind::kPostMove});
    }
  }
  if (eligible.empty()) return {};

  // ---- pick K without replacement (deterministic per game) ----------------
  const int k = std::min<int>(samples_per_game, static_cast<int>(eligible.size()));
  std::mt19937_64 rng(log.seed ^ 0x9E3779B97F4A7C15ULL);  // hash-style salt
  std::vector<int> indices(eligible.size());
  std::iota(indices.begin(), indices.end(), 0);
  // Partial Fisher-Yates: first k entries become the chosen sample.
  for (int i = 0; i < k; ++i) {
    std::uniform_int_distribution<int> dist(i, static_cast<int>(indices.size()) - 1);
    std::swap(indices[i], indices[dist(rng)]);
  }
  std::vector<EligibleRef> chosen;
  chosen.reserve(k);
  for (int i = 0; i < k; ++i) chosen.push_back(eligible[indices[i]]);
  // Sort by (turn_index, kind) so we make a single forward pass through the
  // replay below.
  std::sort(chosen.begin(), chosen.end(), [](const EligibleRef& a, const EligibleRef& b) {
    if (a.turn_index != b.turn_index) return a.turn_index < b.turn_index;
    return static_cast<int>(a.kind) < static_cast<int>(b.kind);
  });

  // ---- pass 2: replay, emit records for chosen positions ------------------
  std::vector<PositionRecord> out;
  out.reserve(k);

  Board board;
  Rack racks[2];
  Move last_move_by[2] = {Move{}, Move{}};  // default PASS
  // Seed each player's rack with their first-turn rack_before.
  bool seeded[2] = {false, false};
  for (const TurnRecord& t : log.turns) {
    if (!seeded[t.player]) {
      racks[t.player] = t.rack_before;
      seeded[t.player] = true;
    }
    if (seeded[0] && seeded[1]) break;
  }

  size_t next_chosen = 0;
  for (size_t i = 0; i < log.turns.size() && next_chosen < chosen.size(); ++i) {
    const TurnRecord& turn = log.turns[i];
    const int active = turn.player;
    const int opp = 1 - active;

    // Sync our replay rack with the recorded rack_before. (Defensive: the
    // log already encodes this; we trust it rather than re-derive from drawn
    // tiles.)
    racks[active] = turn.rack_before;

    // Emit any chosen records at this turn_index.
    while (next_chosen < chosen.size() && chosen[next_chosen].turn_index == static_cast<int>(i)) {
      out.push_back(
        make_record(log, chosen[next_chosen], board, racks[active], racks[opp], last_move_by[opp]));
      ++next_chosen;
    }

    // Advance state: apply the move to board / rack, simulate refill from
    // turn.drawn, simulate exchange-putback (only the bag is affected, which
    // we re-derive on demand so we don't track it here).
    if (turn.move.type == MoveType::PLAY) {
      remove_played_tiles_from_rack(racks[active], turn.move);
      board.apply(turn.move);
    } else if (turn.move.type == MoveType::EXCHANGE) {
      remove_exchanged_tiles_from_rack(racks[active], turn.move);
    }
    // Refill (the log tells us exactly which tiles were drawn).
    for (Tile t : turn.drawn) racks[active].add(t);

    last_move_by[active] = turn.move;
  }

  return out;
}

// ---------------------------------------------------------------------------
// BinaryLogWriter
// ---------------------------------------------------------------------------

BinaryLogWriter::BinaryLogWriter(const std::string& dir, int games_per_file, int samples_per_game)
    : dir_(dir), games_per_file_(games_per_file), samples_per_game_(samples_per_game) {
  if (games_per_file_ < 1) games_per_file_ = 1;
  if (samples_per_game_ < 1) samples_per_game_ = 1;
}

BinaryLogWriter::~BinaryLogWriter() {
  try {
    flush();
  } catch (const std::exception& e) {
    std::cerr << "BinaryLogWriter: error during final flush: " << e.what() << "\n";
  }
}

void BinaryLogWriter::append(const GameLog& log) {
  std::vector<GameLog> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(log);
    if (static_cast<int>(pending_.size()) >= games_per_file_) {
      batch.swap(pending_);
    }
  }
  if (!batch.empty()) write_batch(std::move(batch));
}

void BinaryLogWriter::flush() {
  std::vector<GameLog> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    batch.swap(pending_);
  }
  if (!batch.empty()) write_batch(std::move(batch));
}

void BinaryLogWriter::write_batch(std::vector<GameLog>&& games) {
  // Extract sampled positions for each game.
  std::vector<std::vector<PositionRecord>> per_game;
  per_game.reserve(games.size());
  uint32_t total_positions = 0;
  for (const GameLog& g : games) {
    auto recs = extract_positions(g, samples_per_game_);
    total_positions += static_cast<uint32_t>(recs.size());
    per_game.push_back(std::move(recs));
  }

  // Build metadata table (in-memory; we know all offsets up front).
  const uint64_t header_end = sizeof(FileHeader);
  const uint64_t meta_end = header_end + games.size() * sizeof(GameMetadata);

  std::vector<GameMetadata> meta;
  meta.reserve(games.size());
  uint64_t cursor = meta_end;
  for (size_t i = 0; i < games.size(); ++i) {
    GameMetadata gm{};
    gm.start_offset = cursor;
    gm.num_positions = static_cast<uint32_t>(per_game[i].size());
    gm.data_size = gm.num_positions * static_cast<uint32_t>(sizeof(PositionRecord));
    gm.seed = games[i].seed;
    gm.final_score_p0 = games[i].final_scores[0];
    gm.final_score_p1 = games[i].final_scores[1];
    cursor += gm.data_size;
    meta.push_back(gm);
  }

  // Pick a globally unique filename. The unique_id helper is already used by
  // gcg log filenames.
  std::filesystem::path path =
    std::filesystem::path(dir_) / (std::to_string(get_unique_id()) + ".slog");

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::cerr << "Warning: failed to open binary log file: " << path << "\n";
    return;
  }

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.reserved = 0;
  hdr.num_games = static_cast<uint32_t>(games.size());
  hdr.num_positions = total_positions;
  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(meta.data()),
          static_cast<std::streamsize>(meta.size() * sizeof(GameMetadata)));
  for (const auto& recs : per_game) {
    if (recs.empty()) continue;
    f.write(reinterpret_cast<const char*>(recs.data()),
            static_cast<std::streamsize>(recs.size() * sizeof(PositionRecord)));
  }
  if (!f) {
    std::cerr << "Warning: I/O error writing binary log: " << path << "\n";
  }
}

}  // namespace binlog
}  // namespace scribblez
