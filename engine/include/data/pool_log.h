#pragma once

// The .pool sidecar: the policy-set model's labeled pools
// (docs/plans/deep_bot_v0.md, "The pool record"). A companion of a .slog,
// whose (game_index, turn_index) names each position. Per candidate it keeps
// the move, its labels (sim/pool_labels.h) and its rollouts' win values.
//
// File layout (little-endian, packed):
//
//   [PoolFileHeader                                        128 B]
//   per position:
//     [PoolPositionHeader                                   16 B]
//     [PoolOppRack                                         120 B]
//     [PoolCandidate x num_candidates                       36 B each]
//     [float32 x the sum of the candidates' rollouts: each candidate's
//      rollout win values in turn (rollout_values)]

#include "belief/opp_rack_summary.h"
#include "data/slog_sampling.h"
#include "game/move.h"
#include "sim/pool_labels.h"

#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace scribblez {

// "POOL" in little-endian (bytes 'P','O','O','L' on disk).
inline constexpr uint32_t kPoolMagic = 0x4C4F4F50u;
// The format version (docs/plans/deep_bot_v0.md, "Versions"). Bump it whenever
// the layout or a field's meaning changes; readers accept only this version.
inline constexpr uint16_t kPoolFormatVersion = 5;

// PoolFileHeader::flags bits.
inline constexpr uint16_t kPoolFlagLeaveInference = 1u;  // opponents' leaves came from a model

inline constexpr size_t kPoolHashSize = 16;
inline constexpr size_t kPoolTagSize = 32;

#pragma pack(push, 1)

struct PoolFileHeader {
  uint32_t magic;    // kPoolMagic
  uint16_t version;  // kPoolFormatVersion
  uint16_t flags;    // kPoolFlag* bits
  uint32_t num_positions;
  uint32_t rollouts;  // the budget per candidate; racing stops some early
  // The leaf network every unfinished rollout ended at, after the candidate
  // and two mechanical replies: its ONNX's util::file_content_hash, its tag
  // (NUL-padded), generation and format version. A test's synthetic leaf
  // has no hash.
  char leaf_model_hash[kPoolHashSize];
  char leaf_model_tag[kPoolTagSize];
  int32_t leaf_model_generation;
  int32_t leaf_model_format_version;
  // The leave inference model, all zero under uniform beliefs, likewise.
  char leave_model_hash[kPoolHashSize];
  char leave_model_tag[kPoolTagSize];
  int32_t leave_model_generation;
  int32_t leave_model_format_version;
};
static_assert(sizeof(PoolFileHeader) == 128, "PoolFileHeader must be 128 bytes");

struct PoolPositionHeader {
  uint32_t game_index;  // game within the companion .slog file
  uint32_t turn_index;  // pre-move turn the pool was simmed at
  uint32_t num_candidates;
  uint32_t num_legal_moves;
};
static_assert(sizeof(PoolPositionHeader) == 16, "PoolPositionHeader must be 16 bytes");

// The chooser's belief about the opponent's rack, as the sims drew it
// (belief::OppRackSummary): the policy-set model's input of it.
struct PoolOppRack {
  std::array<float, TILE_KINDS> presence;  // P(the rack holds at least one), by Tile code
  float leave_value_mean;                  // points; 0 unless modelled
  float leave_value_sd;
  uint8_t modelled;  // 1 if the leave came from the leave inference model's belief
  std::array<uint8_t, 3> reserved;
};
static_assert(sizeof(PoolOppRack) == 120, "PoolOppRack must be 120 bytes");

struct PoolCandidate {
  Move move;
  float win_prob;       // PoolLabel::win_prob
  float paired_se;      // PoolLabel::paired_se
  float q;              // PoolLabel::q
  int32_t equity_rank;  // 0-based rank in the static-equity ranking; -1 for a pass outside it
  uint16_t rollouts;    // completed
  uint16_t reserved;    // 0
};
static_assert(sizeof(PoolCandidate) == 36, "PoolCandidate must be 36 bytes");

#pragma pack(pop)

// A model's provenance, as a pool header records it.
struct PoolModel {
  std::string hash;
  std::string tag;
  int generation = 0;
  int format_version = 0;

  bool operator==(const PoolModel&) const = default;
};

// Where a file's labels came from: the leaf network its rollouts ended at,
// and the leave inference model the opponents' leaves came from (empty:
// uniform).
struct PoolSources {
  PoolModel leaf_model;
  PoolModel leave_model;
  int rollouts = 0;

  // The sources a written file's header records.
  static PoolSources of(const PoolFileHeader& h);
  // The name of the first field in which this and `other` differ; nullptr if
  // they agree.
  const char* first_difference(const PoolSources& other) const;
};

// One position's pool, as a writer takes it: parallel per-candidate arrays.
struct PoolPosition {
  binlog::GamePositionIndex at;
  uint32_t num_legal_moves = 0;
  std::vector<Move> moves;
  std::vector<int32_t> equity_ranks;
  std::vector<PoolLabel> labels;
  std::vector<float> rollout_values;  // rollout_values(): labels[c].rollouts per candidate
  belief::OppRackSummary opp_rack;
};

// Accumulates positions in memory and writes the .pool file atomically on
// close().
class PoolWriter {
 public:
  PoolWriter(const std::string& path, const PoolSources& sources);
  // Closes if close() was not called, unless an exception is unwinding: a
  // partial file would pass for a finished one, so it is dropped instead.
  ~PoolWriter();

  PoolWriter(const PoolWriter&) = delete;
  PoolWriter& operator=(const PoolWriter&) = delete;

  void add_position(const PoolPosition& p);
  void close();

 private:
  std::string path_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
  int uncaught_at_open_ = std::uncaught_exceptions();
};

// Loads a whole .pool file and serves per-position views. Throws
// util::Exception on a missing or truncated file, bad magic, or a format
// version other than kPoolFormatVersion.
class PoolReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  struct Position {
    const PoolPositionHeader* header;
    const PoolOppRack* opp_rack;
    const PoolCandidate* candidates;  // header->num_candidates
    // Each candidate's rollout win values in turn, candidates[c].rollouts of
    // them.
    const float* rollout_values;
  };

  explicit PoolReader(const std::string& path);

  const PoolFileHeader& header() const {
    return *reinterpret_cast<const PoolFileHeader*>(buffer_.data());
  }
  int num_positions() const { return int(positions_.size()); }
  Position position(int i) const { return positions_[size_t(i)]; }

 private:
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
