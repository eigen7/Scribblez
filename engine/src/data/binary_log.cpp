#include "data/binary_log.h"

#include "game/bag.h"
#include "game/game_log.h"
#include "util/misc.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace scribblez {
namespace binlog {

namespace {

TurnBlob to_blob(const TurnRecord& t) {
  TurnBlob b{};
  b.move = t.move;
  b.drawn = t.drawn;
  return b;
}

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

GameLog make_game_view(const char* buf, uint32_t game_idx, std::vector<TurnRecord>& scratch,
                       uint32_t* sampled_turn) {
  const GameMetadata* metas = reinterpret_cast<const GameMetadata*>(buf + sizeof(FileHeader));
  const GameMetadata& gm = metas[game_idx];
  const InitialRacks* ir = reinterpret_cast<const InitialRacks*>(buf + gm.start_offset);
  const TurnBlob* turns =
    reinterpret_cast<const TurnBlob*>(buf + gm.start_offset + sizeof(InitialRacks));

  scratch.resize(gm.num_turns);
  for (uint32_t k = 0; k < gm.num_turns; ++k) {
    scratch[k].move = turns[k].move;
    scratch[k].drawn = turns[k].drawn;
  }

  GameLog g;
  g.initial_racks[0] = ir->p0;
  g.initial_racks[1] = ir->p1;
  g.initial_scores = {gm.initial_score_p0, gm.initial_score_p1};
  g.final_scores = {gm.final_score_p0, gm.final_score_p1};
  g.records = scratch.data();
  g.num_records = gm.num_turns;
  if (sampled_turn) *sampled_turn = gm.sampled_turn;
  return g;
}

void complete_turn_records(const GameLog& g, int num_turns, std::vector<TurnRecord>& scratch) {
  std::array<Rack, 2> racks = g.initial_racks;
  std::array<int, 2> scores = g.initial_scores;
  int bag = Bag::kTotalTiles - racks[0].size() - racks[1].size();
  for (int k = 0; k < num_turns; ++k) {
    TurnRecord& rec = scratch[size_t(k)];
    rec.player = k % 2;
    rec.rack_before = racks[rec.player];
    rec.bag_size_before = bag;
    rec.score_delta = rec.move.type() == MoveType::PLAY ? rec.move.score() : 0;
    scores[rec.player] += rec.score_delta;
    rec.cumulative_scores = scores;
    for (int i = 0; i < rec.move.num_glyphs(); ++i)
      racks[rec.player].remove(rec.move.glyph(i).rack_tile());
    // An exchange returns as many tiles as it draws, so only a play drains the bag.
    for (const Tile t : rec.drawn.tiles()) {
      if (t.is_empty()) break;
      racks[rec.player].add(t);
      if (rec.move.type() == MoveType::PLAY) --bag;
    }
  }
}

EligibleSpan eligible_span(const GameLog& log) {
  int end = 0;
  for (int k = 0; k < log.num_records; ++k) {
    if (log.records[k].bag_size_before <= 0) break;
    ++end;
  }
  end = std::min(end, 255);  // GameMetadata's uint8_t field width
  return {std::max(0, log.num_random_opening_plies - 1), end};
}

int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng) {
  const EligibleSpan span = eligible_span(log);
  if (span.begin >= span.end) return -1;
  std::uniform_int_distribution<int> dist(span.begin, span.end - 1);
  return dist(rng);
}

int pick_any_turn(const GameLog& log, std::mt19937_64& rng) {
  if (log.num_records <= 0) return -1;
  std::uniform_int_distribution<int> dist(0, log.num_records - 1);
  return dist(rng);
}

// ---------------------------------------------------------------------------
// BinaryLogWriter
// ---------------------------------------------------------------------------

BinaryLogWriter::BinaryLogWriter(const std::string& dir, int games_per_file, uint16_t flags)
    : dir_(dir), games_per_file_(games_per_file), flags_(flags) {
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
    if (int(pending_.size()) >= games_per_file_) {
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

// A batch of games converted to their on-disk pieces, before file offsets are
// assigned. `games` are views into the source GameLogStorage vector, which must
// outlive the PreparedBatch.
struct PreparedBatch {
  std::vector<InitialRacks> initial;
  std::vector<std::vector<TurnBlob>> turns;
  std::vector<int> sampled_turn;
  std::vector<EligibleSpan> eligible;
  std::vector<GameLog> games;
};

// Drops any game whose eligible region is empty: one that ended during its
// random opening, or whose bag was empty from the first turn.
PreparedBatch prepare_batch(const std::vector<GameLogStorage>& games) {
  PreparedBatch p;
  p.initial.reserve(games.size());
  p.turns.reserve(games.size());
  p.sampled_turn.reserve(games.size());
  p.eligible.reserve(games.size());
  p.games.reserve(games.size());
  std::mt19937_64& rng = sampler_rng();
  for (const GameLogStorage& gs : games) {
    const GameLog g = gs.view();
    const EligibleSpan span = eligible_span(g);
    if (span.begin >= span.end) {
      std::cerr << "BinaryLogWriter: skipping game with no eligible sampling turn\n";
      continue;
    }
    const int sampled = pick_sampled_turn(g, rng);
    p.initial.push_back(initial_racks_of(g));
    std::vector<TurnBlob> turns;
    turns.reserve(size_t(g.num_records));
    for (int k = 0; k < g.num_records; ++k) turns.push_back(to_blob(g.records[k]));
    p.turns.push_back(std::move(turns));
    p.sampled_turn.push_back(sampled);
    p.eligible.push_back(span);
    p.games.push_back(g);
  }
  return p;
}

std::vector<GameMetadata> build_metadata_table(const PreparedBatch& p) {
  std::vector<GameMetadata> meta;
  meta.reserve(p.games.size());
  uint64_t cursor = sizeof(FileHeader) + p.games.size() * sizeof(GameMetadata);
  for (size_t i = 0; i < p.games.size(); ++i) {
    GameMetadata gm{};
    gm.start_offset = cursor;
    gm.num_turns = p.turns[i].size();
    gm.sampled_turn = p.sampled_turn[i];
    gm.final_score_p0 = p.games[i].final_scores[0];
    gm.final_score_p1 = p.games[i].final_scores[1];
    gm.initial_score_p0 = p.games[i].initial_scores[0];
    gm.initial_score_p1 = p.games[i].initial_scores[1];
    gm.eligible_begin = p.eligible[i].begin;
    gm.eligible_end = p.eligible[i].end;
    cursor += sizeof(InitialRacks) + uint64_t(gm.num_turns) * sizeof(TurnBlob);
    meta.push_back(gm);
  }
  return meta;
}

void write_slog_file(const std::filesystem::path& path, const PreparedBatch& p,
                     const std::vector<GameMetadata>& meta, uint16_t flags) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::cerr << "Warning: failed to open binary log file: " << path << "\n";
    return;
  }

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.flags = flags;
  hdr.num_games = p.games.size();
  uint32_t num_sample_positions = 0;
  for (const EligibleSpan& s : p.eligible) num_sample_positions += uint32_t(s.end - s.begin);
  hdr.num_sample_positions = num_sample_positions;
  f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(meta.data()),
          std::streamsize(meta.size() * sizeof(GameMetadata)));
  for (size_t i = 0; i < p.games.size(); ++i) {
    f.write(reinterpret_cast<const char*>(&p.initial[i]), sizeof(InitialRacks));
    const auto& turns = p.turns[i];
    if (!turns.empty()) {
      f.write(reinterpret_cast<const char*>(turns.data()),
              std::streamsize(turns.size() * sizeof(TurnBlob)));
    }
  }
  if (!f) {
    std::cerr << "Warning: I/O error writing binary log: " << path << "\n";
  }
}

}  // namespace

void BinaryLogWriter::write_batch(std::vector<GameLogStorage>&& games) {
  const PreparedBatch prepared = prepare_batch(games);
  if (prepared.games.empty()) return;

  const std::vector<GameMetadata> meta = build_metadata_table(prepared);
  // A strictly increasing timestamp, so no two writes in this process collide.
  const std::filesystem::path path =
    std::filesystem::path(dir_) / std::format("{}.slog", util::get_unique_id());
  write_slog_file(path, prepared, meta, flags_);
}

}  // namespace binlog
}  // namespace scribblez
