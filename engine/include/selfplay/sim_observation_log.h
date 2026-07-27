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
//   [SimObsFileHeader                        16 B]
//   For each position p in [0, num_positions):
//     [SimObsPositionHeader                  24 B]
//     [SimObsRecord   num_candidates(p)    1848 B each]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file. Records store the exact Move alongside its observation, the
// evidence encoding pairing each observation with the move behind it.

#include "game/move.h"
#include "selfplay/sim_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// "SOBS" in little-endian (bytes 'S','O','B','S' on disk).
inline constexpr uint32_t kSimObsMagic = 0x53424F53u;
inline constexpr uint16_t kSimObsVersion = 1;

// SimObsFileHeader::flags bits. Bit 0x1 is RETIRED (it marked sims that used
// the opponent's entire true rack, an information condition no consumer
// supports); readers must reject files carrying it.
inline constexpr uint32_t kSimObsFlagOpenLeaves = 2u;  // sims knew the opponent's retained leave

#pragma pack(push, 1)

struct SimObsFileHeader {
  uint32_t magic;    // kSimObsMagic
  uint16_t version;  // kSimObsVersion
  uint16_t reserved;
  uint32_t num_positions;
  uint32_t flags;  // kSimObsFlag* bits; consumers must match on them
};
static_assert(sizeof(SimObsFileHeader) == 16, "SimObsFileHeader must be 16 bytes");

struct SimObsPositionHeader {
  uint32_t game_index;      // game within the companion .slog file
  uint32_t turn_index;      // pre-move turn the candidates were generated at
  uint32_t num_candidates;  // SimObsRecord count that follows
  uint32_t rollouts;        // rollouts per candidate (== every record's obs.n)
  uint64_t base_seed;       // SimRunner::run seed, for reproducing the sims
};
static_assert(sizeof(SimObsPositionHeader) == 24, "SimObsPositionHeader must be 24 bytes");

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
  explicit SimObsWriter(const std::string& path, uint32_t flags = 0);
  ~SimObsWriter();  // closes if close() was not called

  SimObsWriter(const SimObsWriter&) = delete;
  SimObsWriter& operator=(const SimObsWriter&) = delete;

  // `candidates` and `observations` are parallel arrays; `base_seed` is the
  // SimRunner::run seed used.
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<SimObservation>& observations, uint32_t rollouts,
                    uint64_t base_seed);

  void close();

 private:
  std::string path_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Throws std::runtime_error on a missing file, bad magic, or version mismatch,
// so a stale file fails loudly rather than misparsing.
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

 private:
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
