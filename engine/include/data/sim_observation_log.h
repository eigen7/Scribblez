#pragma once

// The .sobs sidecar format for Monte-Carlo sim observations. One .sobs file
// accompanies one .slog file (binary_log.h). For a subset of that file's
// positions it holds the candidate moves simmed there and each candidate's
// SimObservation (sim/sim_runner.h). Training reads these as the sim-evidence
// inputs described in docs/plans/sim_residual_feedback.md.
//
// Observations are raw counts and moments, never residuals relative to a
// model, so a file stays valid while the proposer model trains. The exception
// is value-truncated sims (docs/roadmap.md item 2): their results depend on
// the leaf evaluator that scores each horizon. The header therefore records
// that model's content hash and the horizon, and consumers must not mix files
// that disagree on them.
//
// File layout
// -----------
//   [SimObsFileHeader                        144 B]
//   For each position p in [0, num_positions):
//     [SimObsPositionHeader                   32 B]
//     [SimObsRecord  x num_candidates(p)   35185 B each]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file.
//
// In a trajectory file (kSimObsFlagTrajectory, docs/roadmap.md item 4) each
// record carries a SimObsRole. Records are stored in sim order: the anchor,
// then on-policy picks, then off-policy draws (see
// training/evidence_trajectory_select.h). Readers must still determine
// evidence eligibility from the role, not from a record's position.

#include "data/sim_obs_role.h"
#include "game/move.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// "SOBS" in little-endian (bytes 'S','O','B','S' on disk).
inline constexpr uint32_t kSimObsMagic = 0x53424F53u;
inline constexpr uint16_t kSimObsVersion = 5;

// SimObsFileHeader::flags bits. Bit 0x1 is reserved and must be rejected by
// readers: it marks sims that saw the opponent's entire true rack, an
// information condition no consumer supports.
inline constexpr uint32_t kSimObsFlagOpenLeaves = 2u;  // sims knew the opponent's retained leave
inline constexpr uint32_t kSimObsFlagTrajectory = 4u;  // record order is trajectory order

// Width of SimObsFileHeader's hex model-content-hash fields, NUL-padded.
// All-zero means no model was involved: the equity-top-K proposer, or
// terminal rollouts.
inline constexpr size_t kSimObsModelHashSize = 64;

#pragma pack(push, 1)

struct SimObsFileHeader {
  uint32_t magic;          // kSimObsMagic
  uint16_t version;        // kSimObsVersion
  uint16_t horizon_plies;  // rollout truncation horizon; 0 = terminal rollouts
  uint32_t num_positions;
  uint32_t flags;  // kSimObsFlag* bits; consumers must match on them
  char proposer_hash[kSimObsModelHashSize];
  char leaf_model_hash[kSimObsModelHashSize];  // all-zero iff horizon_plies == 0
};
static_assert(sizeof(SimObsFileHeader) == 144, "SimObsFileHeader must be 144 bytes");

struct SimObsPositionHeader {
  uint32_t game_index;       // game within the companion .slog file
  uint32_t turn_index;       // pre-move turn the candidates were generated at
  uint32_t num_candidates;   // SimObsRecord count that follows
  uint32_t rollouts;         // rollouts per candidate (== every record's obs.n)
  uint64_t base_seed;        // SimRunner::run seed, for reproducing the sims
  uint32_t num_legal_moves;  // legal moves at the position (the off-policy draws' domain)
  uint32_t flags;            // reserved; always 0
};
static_assert(sizeof(SimObsPositionHeader) == 32, "SimObsPositionHeader must be 32 bytes");

struct SimObsRecord {
  Move move;  // 16 B; the simmed candidate
  SimObservation obs;
  SimObsRole role;  // meaningful only in trajectory files
};
static_assert(sizeof(SimObsRecord) == 16 + sizeof(SimObservation) + 1,
              "SimObsRecord must pack move + observation + role byte with no padding");

#pragma pack(pop)

// Accumulates positions in memory and writes the .sobs file atomically on
// close(), so a partial file never exists on disk.
class SimObsWriter {
 public:
  // `proposer_hash` is the hex content hash of the model that chose the
  // candidates; empty for the equity-top-K proposer. `leaf_model_hash` and
  // `horizon_plies` describe value-truncated sims; pass both, or neither for
  // terminal rollouts. Hashes longer than kSimObsModelHashSize are truncated.
  explicit SimObsWriter(const std::string& path, uint32_t flags = 0,
                        const std::string& proposer_hash = {},
                        const std::string& leaf_model_hash = {}, int horizon_plies = 0);
  ~SimObsWriter();  // closes if close() was not called

  SimObsWriter(const SimObsWriter&) = delete;
  SimObsWriter& operator=(const SimObsWriter&) = delete;

  // `candidates`, `observations` and `roles` are parallel arrays. A
  // non-trajectory writer leaves `roles` empty, and every record then stores
  // kAnchor, which readers of such files ignore.
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<SimObservation>& observations, uint32_t rollouts,
                    uint64_t base_seed, uint32_t num_legal_moves = 0,
                    const std::vector<SimObsRole>& roles = {});

  void close();

 private:
  std::string path_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Loads a whole .sobs file into memory and serves per-position views. Throws
// util::Exception on a missing or truncated file, bad magic, or version
// mismatch. It does not check flags; callers must.
class SimObsReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  struct Position {
    const SimObsPositionHeader* header;
    const SimObsRecord* records;  // header->num_candidates entries
  };

  explicit SimObsReader(const std::string& path);

  int num_positions() const { return positions_.size(); }
  Position position(int i) const { return positions_[i]; }

  uint32_t flags() const { return header().flags; }
  // Empty for the equity-top-K proposer.
  std::string proposer_hash() const;
  // Empty for terminal rollouts.
  std::string leaf_model_hash() const;
  int horizon_plies() const { return header().horizon_plies; }

 private:
  const SimObsFileHeader& header() const {
    return *reinterpret_cast<const SimObsFileHeader*>(buffer_.data());
  }

  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
