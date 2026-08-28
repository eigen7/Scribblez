#pragma once

// Binary sidecar format for Monte-Carlo sim observations. One .sobs file
// accompanies one .slog file (binary_log.h) and holds, for a subset of that
// file's positions, the candidate moves simmed at the position and each
// candidate's SimObservation (sim_runner.h). Training reads these as the
// sim-evidence inputs of docs/sim_residual_feedback.md; positions carry raw
// observations (counts and moments, never model-relative residuals) so the
// file stays valid as the proposer model trains. Value-truncated sims
// (docs/roadmap.md item 2) ARE a function of one model -- the leaf evaluator
// scoring every horizon -- so the header carries that model's content hash
// and the horizon, and consumers must not mix files that disagree on them.
//
// File layout
// -----------
//   [SimObsFileHeader                       144 B]
//   For each position p in [0, num_positions):
//     [SimObsPositionHeader                  32 B]
//     [SimObsRecord   num_candidates(p)    2761 B each]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file. Records store the exact Move alongside its observation, the
// evidence encoding pairing each observation with the move behind it.
//
// In a trajectory file (kSimObsFlagTrajectory) a position's records carry a
// per-record SimObsRole (docs/roadmap.md item 4): the greedy anchor, the
// proposer's on-policy picks, and the stratified off-policy draws. The stored
// order is anchor, then on-policy, then off-policy -- which the sim runner and
// incumbent recovery rely on -- but evidence-eligibility is read off the role,
// not the position, so an off-policy record renders held-out wherever it sits.
// A valid evidence set is any subset of the anchor-plus-on-policy records that
// contains the anchor; off-policy records are labels-only.

#include "data/sim_obs_role.h"
#include "game/move.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// "SOBS" in little-endian (bytes 'S','O','B','S' on disk).
inline constexpr uint32_t kSimObsMagic = 0x53424F53u;
inline constexpr uint16_t kSimObsVersion = 4;

// SimObsFileHeader::flags bits. Bit 0x1 is RETIRED (it marked sims that used
// the opponent's entire true rack, an information condition no consumer
// supports); readers must reject files carrying it.
inline constexpr uint32_t kSimObsFlagOpenLeaves = 2u;  // sims knew the opponent's retained leave
inline constexpr uint32_t kSimObsFlagTrajectory = 4u;  // record order is trajectory order

// SimObsPositionHeader::flags has no bits at v4: the uniform-tail bit it carried
// through v3 is retired -- the off-policy exploration draw is now one of the
// SimObsRole::kOffPolicy records, so held-out-ness is read off the record role.

// Size of SimObsFileHeader's hex model-content-hash fields (the candidate
// proposer, and the truncation leaf evaluator). All-zero bytes mean no such
// model was involved: the equity-top-K proposer, or terminal rollouts.
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
  uint32_t num_legal_moves;  // legal moves at the position (the uniform draw's domain)
  uint32_t flags;            // reserved; no SimObsPosFlag bits at v4
};
static_assert(sizeof(SimObsPositionHeader) == 32, "SimObsPositionHeader must be 32 bytes");

struct SimObsRecord {
  Move move;  // 16 B; the simmed candidate
  SimObservation obs;
  SimObsRole role;  // evidence eligibility; meaningful in trajectory files
};
static_assert(sizeof(SimObsRecord) == 16 + sizeof(SimObservation) + 1,
              "SimObsRecord must pack move + observation + role byte with no padding");

#pragma pack(pop)

// Accumulates positions and writes the .sobs file atomically (temp + rename)
// on close, so nothing exists on disk until then.
class SimObsWriter {
 public:
  // `proposer_hash` is the hex content hash of the model that drove candidate
  // selection; empty for the equity-top-K proposer. `leaf_model_hash` is the
  // hex content hash of the leaf evaluator behind value-truncated sims and
  // `horizon_plies` their horizon; empty/0 for terminal rollouts (give both
  // or neither). Longer hashes truncate to their header fields, matching
  // move_set_eval::TargetWriter.
  explicit SimObsWriter(const std::string& path, uint32_t flags = 0,
                        const std::string& proposer_hash = {},
                        const std::string& leaf_model_hash = {}, int horizon_plies = 0);
  ~SimObsWriter();  // closes if close() was not called

  SimObsWriter(const SimObsWriter&) = delete;
  SimObsWriter& operator=(const SimObsWriter&) = delete;

  // `candidates`, `observations`, and (in a trajectory file) `roles` are
  // parallel arrays; `base_seed` is the SimRunner::run seed used. `roles` is
  // empty for non-trajectory files -- every record then stores kAnchor, which
  // their readers ignore; a trajectory writer passes one role per candidate.
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

  int num_positions() const { return positions_.size(); }
  Position position(int i) const { return positions_[i]; }

  uint32_t flags() const { return header().flags; }
  // Empty for the equity-top-K proposer.
  std::string proposer_hash() const;
  // Empty / 0 for terminal rollouts.
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
