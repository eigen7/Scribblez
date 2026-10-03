#pragma once

// The .sprobe sidecar: SupremeBot M1a's probes (docs/plans/supreme_bot_m1a.md),
// every rollout of every candidate at a position, recorded turn by turn. A
// companion of a .slog, whose (game_index, turn_index) names the position.
//
// File layout (little-endian, packed):
//
//   [ProbeFileHeader                                          96 B]
//   per position:
//     [ProbePositionHeader                                    32 B]
//     [ProbeCandidate   x num_candidates                      28 B each]
//     [ProbeRecord      x num_candidates * probes             44 B each]
//     [binlog::TurnBlob x num_turns                           24 B each]
//
// Records are candidate-major: candidate c's probe i is record
// c * probes + i, and its turns follow those of every record before it.
//
// Like a .slog, a record keeps only what replay cannot recompute: the racks
// each side held when the rollout began, and each turn's move and draw. A
// turn's mover, rack, bag count and score follow by replaying from the
// position after the candidate. The outcome is kept, since a truncated
// rollout's comes from the leaf model.

#include "data/binary_log.h"
#include "data/slog_sampling.h"
#include "game/move.h"
#include "game/rack.h"
#include "sim/sim_runner.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {

// "SPRB" in little-endian (bytes 'S','P','R','B' on disk).
inline constexpr uint32_t kProbeMagic = 0x42525053u;
inline constexpr uint16_t kProbeVersion = 1;

// ProbeFileHeader::flags bits.
inline constexpr uint16_t kProbeFlagFaceUpLeaves = 1u;  // probes knew the opponent's leave

inline constexpr size_t kProbeModelHashSize = 64;
inline constexpr size_t kProbeLexiconSize = 16;

#pragma pack(push, 1)

struct ProbeFileHeader {
  uint32_t magic;    // kProbeMagic
  uint16_t version;  // kProbeVersion
  uint16_t flags;    // kProbeFlag* bits
  uint32_t num_positions;
  uint16_t horizon_plies;  // plies after the candidate before the leaf model scores
  uint16_t probes;         // probes per candidate
  char leaf_model_hash[kProbeModelHashSize];
  char lexicon[kProbeLexiconSize];  // the lexicon's name, NUL-padded
};
static_assert(sizeof(ProbeFileHeader) == 96, "ProbeFileHeader must be 96 bytes");

struct ProbePositionHeader {
  uint32_t game_index;  // game within the companion .slog file
  uint32_t turn_index;  // pre-move turn the candidates were generated at
  uint64_t base_seed;   // probe i of every candidate is seeded by base_seed + i
  uint32_t num_candidates;
  uint32_t num_legal_moves;
  uint32_t num_turns;  // TurnBlobs that follow the records
  uint32_t reserved;   // 0
};
static_assert(sizeof(ProbePositionHeader) == 32, "ProbePositionHeader must be 32 bytes");

struct ProbeCandidate {
  Move move;
  float equity;         // HastyBot static equity
  int32_t equity_rank;  // 0-based rank in the static-equity ranking
  uint8_t stratum;      // a Stratum (sim/transfer_candidates.h)
  uint8_t reserved[3];  // 0
};
static_assert(sizeof(ProbeCandidate) == 28, "ProbeCandidate must be 28 bytes");

struct ProbeRecord {
  Rack mover_rack;  // the root mover's leave plus refill when the rollout began
  Rack opp_rack;    // the opponent's rack it replied from
  // The outcome, from the root mover's point of view: 0/1 and the exact delta
  // when the game ended, the leaf model's readings when truncated.
  float p_win;
  float p_draw;
  float p_loss;
  float delta;
  float delta_sq;
  uint16_t candidate;
  uint16_t probe;  // the rollout's index among the candidate's probes
  uint8_t truncated;
  uint8_t num_turns;
  uint8_t reserved[2];  // 0
};
static_assert(sizeof(ProbeRecord) == 44, "ProbeRecord must be 44 bytes");

#pragma pack(pop)

// One position's probes, as a writer takes them: parallel per-candidate arrays,
// and each candidate's rollouts with their traces (SimOutput::kTraces).
struct ProbePosition {
  binlog::GamePositionIndex at;
  int mover = 0;  // the root mover, whose candidates these are
  uint64_t base_seed = 0;
  uint32_t num_legal_moves = 0;
  std::vector<Move> moves;
  std::vector<float> equities;
  std::vector<int32_t> equity_ranks;
  std::vector<uint8_t> strata;
  std::vector<std::vector<Rollout>> rollouts;     // rollouts[c][i]
  std::vector<std::vector<RolloutTrace>> traces;  // traces[c][i]
};

// Accumulates positions in memory and writes the .sprobe file atomically on
// close().
class ProbeWriter {
 public:
  ProbeWriter(const std::string& path, uint16_t flags, const std::string& leaf_model_hash,
              const std::string& lexicon, int horizon_plies, int probes);
  ~ProbeWriter();  // closes if close() was not called

  ProbeWriter(const ProbeWriter&) = delete;
  ProbeWriter& operator=(const ProbeWriter&) = delete;

  // Every candidate must have exactly the writer's probe count.
  void add_position(const ProbePosition& p);
  void close();

 private:
  std::string path_;
  int probes_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Loads a whole .sprobe file and serves per-position views. Throws
// util::Exception on a missing or truncated file, bad magic, or version
// mismatch.
class ProbeReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  struct Position {
    const ProbePositionHeader* header;
    const ProbeCandidate* candidates;  // header->num_candidates
    const ProbeRecord* records;        // header->num_candidates * probes()
    const binlog::TurnBlob* turns;     // header->num_turns, in record order
  };

  explicit ProbeReader(const std::string& path);

  const ProbeFileHeader& header() const {
    return *reinterpret_cast<const ProbeFileHeader*>(buffer_.data());
  }
  int probes() const { return header().probes; }
  int num_positions() const { return int(positions_.size()); }
  Position position(int i) const { return positions_[size_t(i)]; }

 private:
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
