#include "scribblez/binary_log.h"

#include "scribblez/game.h"
#include "scribblez/unique_id.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

namespace scribblez {
namespace binlog {

namespace {

// Pack a TurnRecord into a TurnBlob: copy the move and the post-turn draws.
TurnBlob to_blob(const TurnRecord& t) {
  TurnBlob b{};
  b.move = t.move;
  b.drawn = t.drawn;
  b.num_drawn = t.num_drawn;
  return b;
}

// Pack a GameLog's initial racks into an InitialRacks blob.
InitialRacks initial_racks_of(const GameLog& log) {
  InitialRacks ir{};
  ir.p0 = log.initial_racks[0].tiles();
  ir.p1 = log.initial_racks[1].tiles();
  ir.n0 = static_cast<uint8_t>(log.initial_racks[0].size());
  ir.n1 = static_cast<uint8_t>(log.initial_racks[1].size());
  return ir;
}

}  // namespace

// ---------------------------------------------------------------------------
// BinaryLogWriter
// ---------------------------------------------------------------------------

BinaryLogWriter::BinaryLogWriter(const std::string& dir, int games_per_file)
    : dir_(dir), games_per_file_(games_per_file) {
  if (games_per_file_ < 1) games_per_file_ = 1;
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
  // Pre-build per-game blobs so we can compute offsets up front.
  std::vector<InitialRacks> per_game_initial;
  std::vector<std::vector<TurnBlob>> per_game_turns;
  per_game_initial.reserve(games.size());
  per_game_turns.reserve(games.size());
  uint32_t total_positions = 0;
  for (const GameLog& g : games) {
    per_game_initial.push_back(initial_racks_of(g));
    std::vector<TurnBlob> turns;
    turns.reserve(g.turns.size());
    for (const TurnRecord& t : g.turns) turns.push_back(to_blob(t));
    per_game_turns.push_back(std::move(turns));
    total_positions += static_cast<uint32_t>(g.turns.size());
  }

  // Build metadata table (in-memory; we know all offsets up front).
  const uint64_t meta_end = sizeof(FileHeader) + games.size() * sizeof(GameMetadata);

  std::vector<GameMetadata> meta;
  meta.reserve(games.size());
  uint64_t cursor = meta_end;
  for (size_t i = 0; i < games.size(); ++i) {
    GameMetadata gm{};
    gm.start_offset = cursor;
    gm.num_turns = static_cast<uint32_t>(per_game_turns[i].size());
    gm.reserved = 0;
    gm.final_score_p0 = games[i].final_scores[0];
    gm.final_score_p1 = games[i].final_scores[1];
    cursor += sizeof(InitialRacks) + static_cast<uint64_t>(gm.num_turns) * sizeof(TurnBlob);
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
  for (size_t i = 0; i < games.size(); ++i) {
    f.write(reinterpret_cast<const char*>(&per_game_initial[i]), sizeof(InitialRacks));
    const auto& turns = per_game_turns[i];
    if (!turns.empty()) {
      f.write(reinterpret_cast<const char*>(turns.data()),
              static_cast<std::streamsize>(turns.size() * sizeof(TurnBlob)));
    }
  }
  if (!f) {
    std::cerr << "Warning: I/O error writing binary log: " << path << "\n";
  }
}

}  // namespace binlog
}  // namespace scribblez
