#pragma once

// Binary sidecar format for Monte-Carlo sim observations. One .sobs file
// accompanies one .slog file (binary_log.h) and holds, for a subset of that
// file's positions, the candidate moves simmed at the position and each
// candidate's SimObservation (sim_runner.h). Training reads these as the
// sim-evidence inputs of docs/sim_residual_feedback.md; positions carry raw
// observations (counts and moments, never model-relative residuals) so the
// file stays valid as the model trains.
//
// File layout
// -----------
//   [SimObsFileHeader                        80 B]
//   For each position p in [0, num_positions):
//     [SimObsPositionHeader                  32 B]
//     [SimObsRecord   num_candidates(p)    1848 B each]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file. Records store the exact Move alongside its observation, the
// evidence encoding pairing each observation with the move behind it.
//
// In a trajectory file (kSimObsFlagTrajectory) a position's records are in
// the order the sequential loop simmed them -- the greedy anchor first, then
// the proposer's picks -- so every prefix of the record array is a valid
// evidence set. A trailing uniform-exploration draw, when present, is marked
// by kSimObsPosFlagUniformTail (it can be absent: a small legal set may be
// exhausted before the uniform slot).

#include "game/move.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// "SOBS" in little-endian (bytes 'S','O','B','S' on disk).
inline constexpr uint32_t kSimObsMagic = 0x53424F53u;
inline constexpr uint16_t kSimObsVersion = 2;

// SimObsFileHeader::flags bits. Bit 0x1 is RETIRED (it marked sims that used
// the opponent's entire true rack, an information condition no consumer
// supports); readers must reject files carrying it.
inline constexpr uint32_t kSimObsFlagOpenLeaves = 2u;  // sims knew the opponent's retained leave
inline constexpr uint32_t kSimObsFlagTrajectory = 4u;  // record order is trajectory order

// SimObsPositionHeader::flags bits.
inline constexpr uint32_t kSimObsPosFlagUniformTail = 1u;  // last record is the uniform draw

// Hex content hash of the proposer model, in SimObsFileHeader. All-zero bytes
// mean no model drove the candidate selection (the equity-top-K proposer).
inline constexpr size_t kSimObsProposerHashSize = 64;

#pragma pack(push, 1)

struct SimObsFileHeader {
  uint32_t magic;    // kSimObsMagic
  uint16_t version;  // kSimObsVersion
  uint16_t reserved;
  uint32_t num_positions;
  uint32_t flags;  // kSimObsFlag* bits; consumers must match on them
  char proposer_hash[kSimObsProposerHashSize];
};
static_assert(sizeof(SimObsFileHeader) == 80, "SimObsFileHeader must be 80 bytes");

struct SimObsPositionHeader {
  uint32_t game_index;       // game within the companion .slog file
  uint32_t turn_index;       // pre-move turn the candidates were generated at
  uint32_t num_candidates;   // SimObsRecord count that follows
  uint32_t rollouts;         // rollouts per candidate (== every record's obs.n)
  uint64_t base_seed;        // SimRunner::run seed, for reproducing the sims
  uint32_t num_legal_moves;  // legal moves at the position (the uniform draw's domain)
  uint32_t flags;            // kSimObsPosFlag* bits
};
static_assert(sizeof(SimObsPositionHeader) == 32, "SimObsPositionHeader must be 32 bytes");

struct SimObsRecord {
  Move move;  // 16 B; the simmed candidate
  SimObservation obs;
};
static_assert(sizeof(SimObsRecord) == 16 + sizeof(SimObservation),
              "SimObsRecord must pack move + observation with no padding");

#pragma pack(pop)

// Accumulates positions and writes the .sobs file atomically (temp + rename)
// on close, so nothing exists on disk until then.
class SimObsWriter {
 public:
  // `proposer_hash` is the hex content hash of the model that drove candidate
  // selection; empty for the equity-top-K proposer. Longer hashes truncate to
  // the header field, matching move_set_eval::TargetWriter.
  explicit SimObsWriter(const std::string& path, uint32_t flags = 0,
                        const std::string& proposer_hash = {});
  ~SimObsWriter();  // closes if close() was not called

  SimObsWriter(const SimObsWriter&) = delete;
  SimObsWriter& operator=(const SimObsWriter&) = delete;

  // `candidates` and `observations` are parallel arrays; `base_seed` is the
  // SimRunner::run seed used.
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<SimObservation>& observations, uint32_t rollouts,
                    uint64_t base_seed, uint32_t num_legal_moves = 0, uint32_t position_flags = 0);

  void close();

 private:
  std::string path_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Loads a .sobs file into memory and serves per-position views. Throws
// util::Exception on a missing file, bad magic, or version mismatch, so a
// stale file fails loudly rather than misparsing.
class SimObsReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  struct Position {
    const SimObsPositionHeader* header;
    const SimObsRecord* records;  // header->num_candidates entries
  };

  explicit SimObsReader(const std::string& path);

  int num_positions() const { return static_cast<int>(positions_.size()); }
  Position position(int i) const { return positions_[i]; }

  uint32_t flags() const { return header().flags; }
  // Empty for the equity-top-K proposer.
  std::string proposer_hash() const;

 private:
  const SimObsFileHeader& header() const {
    return *reinterpret_cast<const SimObsFileHeader*>(buffer_.data());
  }

  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
