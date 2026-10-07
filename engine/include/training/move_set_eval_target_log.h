#pragma once

// The .mset sidecar: distillation targets for the move set evaluation model.
// One .mset accompanies one .slog and holds, for a subset of its positions, the
// candidate moves selected there (move_set_eval_candidates.h) and the teacher
// position evaluation model's readouts at each candidate's post-move state. The
// student trainer pairs these with inputs it reconstructs by replaying the
// .slog (docs/architecture.md).
//
// The header pins the teacher by content hash, so a corpus can be verified to
// come from one checkpoint. It also records the per-record target widths, so
// heads can be appended without restructuring the format.
//
// File layout
// -----------
//   [TargetFileHeader                              88 B]
//   For each position p in [0, num_positions):
//     [TargetPositionHeader                        16 B]
//     [num_candidates(p) records, each:
//        Move                                      16 B
//        record_floats x float                     value targets
//        record_planes x (float + kPlaneWidth B)   legacy plane block]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file, addressing the PRE-move decision point; each record's targets
// describe the post-move state its Move produces.
//
// The writer emits record_planes = 0: the student distills the value heads
// only. Files written before that carry the teacher's four quantized
// footprint distributions per record (record_planes = 4: four float scales,
// then four kPlaneWidth-byte planes), which readers step over.
//
// A file holds one kind of position throughout, declared by
// kTargetFlagFullSweep: the stratified training sample or the full-sweep
// evaluation slice. Swept positions are held out, and the trainer routes
// train versus held-out by file, so the two never mix.

#include "game/move.h"
#include "training/footprint.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// "MSET" in little-endian (bytes 'M','S','E','T' on disk).
inline constexpr uint32_t kTargetMagic = 0x5445534Du;
inline constexpr uint16_t kTargetVersion = 3;
// The value-target floats per candidate record, mover POV, in record order.
inline constexpr std::array<const char*, 5> kTargetNamesV1 = {"p_win", "p_draw", "p_loss",
                                                              "sd_mean", "sd_std"};
inline constexpr uint32_t kTargetFloatsV1 = kTargetNamesV1.size();
// Bytes per legacy plane (see the file layout): one per footprint class.
inline constexpr uint32_t kPlaneWidth = kFootprintClasses;

// TargetFileHeader::flags bits, mirroring the .sobs convention.
inline constexpr uint32_t kTargetFlagOpenLeaves = 2u;
// Every position is a (capped) full sweep of its legal candidates rather than a
// stratified sample. The trainer holds such a file out and never trains on it.
inline constexpr uint32_t kTargetFlagFullSweep = 4u;

// FP16 teacher inference can overflow the score-diff std readout to +inf on
// near-terminal states: the exported graph evaluates Softplus naively in half
// precision, which overflows for logits above ~11 although the true value
// there is about the logit itself. The generator stores the clamped std so
// every record is finite. The WLD targets, which matter most, are unaffected,
// and the std head is extrapolating on such states anyway. The cap sits well
// above genuine teacher readouts (< ~90 across the shakeout corpus).
inline constexpr float kSdStdCap = 128.0f;

inline float clamped_sd_std(float sd_std) { return std::min(sd_std, kSdStdCap); }

inline constexpr int kTargetModelHashChars = 64;

// The .mset flags for a .slog with FileHeader flags `slog_flags`: the
// information condition the games were played under, which the trainer must
// match with the student's input arm.
uint32_t target_flags_from_slog(uint16_t slog_flags);

#pragma pack(push, 1)

struct TargetFileHeader {
  uint32_t magic;    // kTargetMagic
  uint16_t version;  // kTargetVersion
  uint16_t reserved;
  uint32_t num_positions;
  uint32_t record_floats;                  // target floats per candidate record
  uint32_t record_planes;                  // legacy plane blocks per record (written 0)
  uint32_t flags;                          // kTargetFlag* bits
  char model_hash[kTargetModelHashChars];  // hex, NUL-padded
};
static_assert(sizeof(TargetFileHeader) == 88, "TargetFileHeader must be 88 bytes");

struct TargetPositionHeader {
  uint32_t game_index;      // game within the companion .slog file
  uint32_t turn_index;      // pre-move turn the candidates were sampled at
  uint32_t num_candidates;  // records that follow
  // For a swept position, its legal-move count, so a sweep truncated by the
  // cap shows as num_candidates < num_legal_moves. 0 for a stratified
  // position, where it is not recorded.
  uint32_t num_legal_moves;
};
static_assert(sizeof(TargetPositionHeader) == 16, "TargetPositionHeader must be 16 bytes");

#pragma pack(pop)

// Accumulates positions in memory and writes the .mset on close. The write is
// atomic, so an interrupted run never leaves a truncated file that a resume,
// which skips existing sidecars, would silently keep.
class TargetWriter {
 public:
  TargetWriter(const std::string& path, uint32_t record_floats, const std::string& model_hash,
               uint32_t flags = 0);
  // Closes if close() was not called, unless an exception is unwinding: a
  // partial file would pass for a finished one, so it is dropped instead.
  ~TargetWriter();

  TargetWriter(const TargetWriter&) = delete;
  TargetWriter& operator=(const TargetWriter&) = delete;

  // `targets` is candidates.size() x record_floats, candidate-major.
  // `num_legal_moves` as in TargetPositionHeader.
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<float>& targets, uint32_t num_legal_moves = 0);

  void close();

 private:
  std::string path_;
  uint32_t record_floats_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
  int uncaught_at_open_ = std::uncaught_exceptions();
};

// Loads a .mset file into memory and serves per-position views. Throws
// util::Exception on a missing or truncated file, bad magic, or a version
// mismatch, so a stale file fails loudly rather than misparsing.
class TargetReader {
 public:
  // A view into the reader's buffer, valid while the reader lives. Read
  // `records` through the accessors below.
  struct Position {
    const TargetPositionHeader* header;
    const char* records;
  };

  explicit TargetReader(const std::string& path);

  uint32_t record_floats() const { return header_.record_floats; }
  uint32_t record_planes() const { return header_.record_planes; }
  uint32_t flags() const { return header_.flags; }
  std::string model_hash() const;
  int num_positions() const { return positions_.size(); }
  Position position(int i) const { return positions_[i]; }

  Move move_at(const Position& p, int candidate) const;
  const float* targets_at(const Position& p, int candidate) const;

 private:
  size_t record_bytes() const {
    return sizeof(Move) + sizeof(float) * header_.record_floats +
           header_.record_planes * (sizeof(float) + kPlaneWidth);
  }

  TargetFileHeader header_{};
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace move_set_eval
}  // namespace scribblez
