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

// Count the training-eligible turns of `log`. When `include_endgame` is false,
// only pre-endgame turns (bag had tiles when the turn began) are eligible; when
// true, every turn is. Because the bag is non-increasing across turns, the
// eligible turns form a leading prefix [0, count), so a single count fully
// describes the eligible set (recorded as GameMetadata::eligible_turns and
// expanded into one training row per eligible turn at load time).
int eligible_turn_count(const GameLog& log, bool include_endgame) {
  if (include_endgame) return log.num_records;
  int count = 0;
  for (int k = 0; k < log.num_records; ++k) {
    if (log.records[k].bag_size_before <= 0) break;  // prefix ends at the first endgame turn
    ++count;
  }
  return count;
}

std::mt19937_64& sampler_rng() {
  thread_local std::mt19937_64 rng(std::random_device{}());
  return rng;
}

}  // namespace

int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng, bool include_endgame) {
  std::vector<int> eligible;
  eligible.reserve(static_cast<size_t>(log.num_records));
  for (int k = 0; k < log.num_records; ++k) {
    if (include_endgame || log.records[k].bag_size_before > 0) eligible.push_back(k);
  }
  if (eligible.empty()) return -1;
  std::uniform_int_distribution<size_t> dist(0, eligible.size() - 1);
  return eligible[dist(rng)];
}

// ---------------------------------------------------------------------------
// BinaryLogWriter
// ---------------------------------------------------------------------------

BinaryLogWriter::BinaryLogWriter(const std::string& dir, int games_per_file, bool sample_endgames)
    : dir_(dir), games_per_file_(games_per_file), sample_endgames_(sample_endgames) {
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

namespace {

// One batch of games packed into serialization-ready pieces, gathered before
// any file offsets are known. The GameLog entries are non-owning views into the
// source GameLogStorage vector, which must outlive the PreparedBatch.
struct PreparedBatch {
  std::vector<InitialRacks> initial;
  std::vector<std::vector<TurnBlob>> turns;
  std::vector<int> sampled_turn;  // eval-only representative position per game
  std::vector<int> eligible;      // training-eligible turn count per game (prefix)
  std::vector<GameLog> games;
};

// Pre-build per-game blobs. `include_endgame` controls whether post-bag-empty
// positions are eligible. Each kept game records its eligible-turn count (the
// prefix [0, eligible) training expands over) and one eval-only sampled turn
// drawn uniformly from that prefix. Games with no eligible turn (bag empty for
// every turn with endgame sampling off -- shouldn't happen in practice but we
// guard anyway) are dropped.
PreparedBatch prepare_batch(const std::vector<GameLogStorage>& games, bool include_endgame) {
  PreparedBatch p;
  p.initial.reserve(games.size());
  p.turns.reserve(games.size());
  p.sampled_turn.reserve(games.size());
  p.eligible.reserve(games.size());
  p.games.reserve(games.size());
  std::mt19937_64& rng = sampler_rng();
  for (const GameLogStorage& gs : games) {
    const GameLog g = gs.view();
    const int eligible = eligible_turn_count(g, include_endgame);
    if (eligible <= 0) {
      std::cerr << "BinaryLogWriter: skipping game with no eligible sampling turn\n";
      continue;
    }
    const int sampled = pick_sampled_turn(g, rng, include_endgame);
    p.initial.push_back(initial_racks_of(g));
    std::vector<TurnBlob> turns;
    turns.reserve(static_cast<size_t>(g.num_records));
    for (int k = 0; k < g.num_records; ++k) turns.push_back(to_blob(g.records[k]));
    p.turns.push_back(std::move(turns));
    p.sampled_turn.push_back(sampled);
    p.eligible.push_back(eligible);
    p.games.push_back(g);
  }
  return p;
}

// Build the metadata index: one entry per game with its absolute start offset
// (all offsets are known up front) plus scores and turn counts.
std::vector<GameMetadata> build_metadata_table(const PreparedBatch& p) {
  std::vector<GameMetadata> meta;
  meta.reserve(p.games.size());
  uint64_t cursor = sizeof(FileHeader) + p.games.size() * sizeof(GameMetadata);
  for (size_t i = 0; i < p.games.size(); ++i) {
    GameMetadata gm{};
    gm.start_offset = cursor;
    gm.num_turns = static_cast<uint32_t>(p.turns[i].size());
    gm.sampled_turn = static_cast<uint32_t>(p.sampled_turn[i]);
    gm.final_score_p0 = static_cast<int16_t>(p.games[i].final_scores[0]);
    gm.final_score_p1 = static_cast<int16_t>(p.games[i].final_scores[1]);
    gm.eligible_turns = static_cast<uint16_t>(p.eligible[i]);
    cursor += sizeof(InitialRacks) + static_cast<uint64_t>(gm.num_turns) * sizeof(TurnBlob);
    meta.push_back(gm);
  }
  return meta;
}

// Write header, metadata table, and per-game data to path.
void write_slog_file(const std::filesystem::path& path, const PreparedBatch& p,
                     const std::vector<GameMetadata>& meta) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::cerr << "Warning: failed to open binary log file: " << path << "\n";
    return;
  }

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.reserved = 0;
  hdr.num_games = static_cast<uint32_t>(p.games.size());
  uint32_t num_sample_positions = 0;
  for (int e : p.eligible) num_sample_positions += static_cast<uint32_t>(e);
  hdr.num_sample_positions = num_sample_positions;
  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(meta.data()),
          static_cast<std::streamsize>(meta.size() * sizeof(GameMetadata)));
  for (size_t i = 0; i < p.games.size(); ++i) {
    f.write(reinterpret_cast<const char*>(&p.initial[i]), sizeof(InitialRacks));
    const auto& turns = p.turns[i];
    if (!turns.empty()) {
      f.write(reinterpret_cast<const char*>(turns.data()),
              static_cast<std::streamsize>(turns.size() * sizeof(TurnBlob)));
    }
  }
  if (!f) {
    std::cerr << "Warning: I/O error writing binary log: " << path << "\n";
  }
}

}  // namespace

void BinaryLogWriter::write_batch(std::vector<GameLogStorage>&& games) {
  const PreparedBatch prepared = prepare_batch(games, sample_endgames_);
  if (prepared.games.empty()) return;

  const std::vector<GameMetadata> meta = build_metadata_table(prepared);
  // unique_id keeps slog filenames globally unique, as with gcg log filenames.
  const std::filesystem::path path =
    std::filesystem::path(dir_) / (std::to_string(get_unique_id()) + ".slog");
  write_slog_file(path, prepared, meta);
}

}  // namespace binlog
}  // namespace scribblez
