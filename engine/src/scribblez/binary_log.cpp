#include "scribblez/binary_log.h"

#include "scribblez/game.h"
#include "scribblez/unique_id.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace scribblez {
namespace binlog {

namespace {

// Pack a TurnRecord into a TurnBlob: copy the move and the post-turn draws.
TurnBlob to_blob(const TurnRecord& t) {
  TurnBlob b{};
  b.move = t.move;
  b.drawn = t.drawn;
  return b;
}

// Pack a GameLog's initial racks into an InitialRacks blob.
InitialRacks initial_racks_of(const GameLog& log) {
  InitialRacks ir{};
  ir.p0 = log.initial_racks[0];
  ir.p1 = log.initial_racks[1];
  return ir;
}

std::mt19937_64& sampler_rng() {
  thread_local std::mt19937_64 rng(std::random_device{}());
  return rng;
}

}  // namespace

int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng) {
  std::vector<int> eligible;
  eligible.reserve(static_cast<size_t>(log.num_records));
  for (int k = 0; k < log.num_records; ++k) {
    if (log.records[k].bag_size_before > 0) eligible.push_back(k);
  }
  if (eligible.empty()) return -1;
  std::uniform_int_distribution<size_t> dist(0, eligible.size() - 1);
  return eligible[dist(rng)];
}

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

void BinaryLogWriter::append(GameLogStorage&& log) {
  std::vector<GameLogStorage> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.push_back(std::move(log));
    if (static_cast<int>(pending_.size()) >= games_per_file_) {
      batch.swap(pending_);
    }
  }
  if (!batch.empty()) write_batch(std::move(batch));
}

void BinaryLogWriter::flush() {
  std::vector<GameLogStorage> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    batch.swap(pending_);
  }
  if (!batch.empty()) write_batch(std::move(batch));
}

void BinaryLogWriter::write_batch(std::vector<GameLogStorage>&& games) {
  // Pre-build per-game blobs so we can compute offsets up front. Games
  // with no eligible sampling turn (bag empty for every turn -- shouldn't
  // happen in practice but we guard anyway) are dropped here.
  std::vector<InitialRacks> per_game_initial;
  std::vector<std::vector<TurnBlob>> per_game_turns;
  std::vector<int> per_game_sampled_turn;
  std::vector<GameLog> kept_games;  // non-owning views into `games`
  per_game_initial.reserve(games.size());
  per_game_turns.reserve(games.size());
  per_game_sampled_turn.reserve(games.size());
  kept_games.reserve(games.size());
  std::mt19937_64& rng = sampler_rng();
  for (const GameLogStorage& gs : games) {
    const GameLog g = gs.view();
    const int sampled = pick_sampled_turn(g, rng);
    if (sampled < 0) {
      std::cerr << "BinaryLogWriter: skipping game with no eligible sampling turn\n";
      continue;
    }
    per_game_initial.push_back(initial_racks_of(g));
    std::vector<TurnBlob> turns;
    turns.reserve(static_cast<size_t>(g.num_records));
    for (int k = 0; k < g.num_records; ++k) turns.push_back(to_blob(g.records[k]));
    per_game_turns.push_back(std::move(turns));
    per_game_sampled_turn.push_back(sampled);
    kept_games.push_back(g);
  }
  if (kept_games.empty()) return;

  // Build metadata table (in-memory; we know all offsets up front).
  const uint64_t meta_end = sizeof(FileHeader) + kept_games.size() * sizeof(GameMetadata);

  std::vector<GameMetadata> meta;
  meta.reserve(kept_games.size());
  uint64_t cursor = meta_end;
  for (size_t i = 0; i < kept_games.size(); ++i) {
    GameMetadata gm{};
    gm.start_offset = cursor;
    gm.num_turns = static_cast<uint32_t>(per_game_turns[i].size());
    gm.sampled_turn = static_cast<uint32_t>(per_game_sampled_turn[i]);
    gm.final_score_p0 = static_cast<int16_t>(kept_games[i].final_scores[0]);
    gm.final_score_p1 = static_cast<int16_t>(kept_games[i].final_scores[1]);
    gm.initial_score_p0 = static_cast<int16_t>(kept_games[i].initial_scores[0]);
    gm.initial_score_p1 = static_cast<int16_t>(kept_games[i].initial_scores[1]);
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
  hdr.num_games = static_cast<uint32_t>(kept_games.size());
  hdr.reserved2 = 0;
  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(meta.data()),
          static_cast<std::streamsize>(meta.size() * sizeof(GameMetadata)));
  for (size_t i = 0; i < kept_games.size(); ++i) {
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
