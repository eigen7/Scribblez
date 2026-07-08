#pragma once

// Binary sidecar format for move set evaluation model distillation targets.
// One .mset file accompanies one .slog file (binary_log.h) and holds, for a
// subset of that file's positions, the candidate moves sampled at the
// position and the teacher position evaluation model's readouts for each
// candidate's post-move state. The move set trainer pairs these targets with
// inputs it reconstructs by replay (docs/roadmap2.md, track A).
//
// The teacher is pinned: the header records the content hash of the teacher
// ONNX that produced the targets, so a corpus can be verified to come from
// one blessed checkpoint. The per-record target width is also in the header
// (`record_floats`), making the record layout head-extensible: readers
// consume the widths they know and a future version can append heads without
// format surgery.
//
// File layout
// -----------
//   [TargetFileHeader                              84 B]
//   For each position p in [0, num_positions):
//     [TargetPositionHeader                        16 B]
//     [num_candidates(p) x (Move 16 B + record_floats x float)]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file, addressing the PRE-move decision point; each record's targets
// describe the post-move state its Move produces. Version-1 targets, in
// order: [p_win, p_draw, p_loss, score_diff_mean, score_diff_std], all from
// the mover's POV (kTargetFloatsV1).

#include "scribblez/move.h"

#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// "MSET" in little-endian (bytes 'M','S','E','T' on disk).
inline constexpr uint32_t kTargetMagic = 0x5445534Du;
inline constexpr uint16_t kTargetVersion = 1;
inline constexpr uint32_t kTargetFloatsV1 = 5;  // [p_win, p_draw, p_loss, sd_mean, sd_std]

// TargetFileHeader::flags bits. Mirrors the .sobs convention
// (sim_observation_log.h): bit 0x2 marks the open-leaves information
// condition (inputs carry the opponent-leave block).
inline constexpr uint32_t kTargetFlagOpenLeaves = 2u;

inline constexpr int kTargetModelHashChars = 64;

#pragma pack(push, 1)

struct TargetFileHeader {
  uint32_t magic;    // kTargetMagic
  uint16_t version;  // kTargetVersion
  uint16_t reserved;
  uint32_t num_positions;
  uint32_t record_floats;  // target floats per candidate record
  uint32_t flags;          // kTargetFlag* bits
  // Hex content hash of the teacher ONNX (NUL-padded), pinning the corpus to
  // one checkpoint.
  char model_hash[kTargetModelHashChars];
};
static_assert(sizeof(TargetFileHeader) == 84, "TargetFileHeader must be 84 bytes");

struct TargetPositionHeader {
  uint32_t game_index;      // game within the companion .slog file
  uint32_t turn_index;      // pre-move turn the candidates were sampled at
  uint32_t num_candidates;  // records that follow
  uint32_t reserved;
};
static_assert(sizeof(TargetPositionHeader) == 16, "TargetPositionHeader must be 16 bytes");

#pragma pack(pop)

// Streams positions to a .mset file. Buffered in memory and written atomically
// (temp + rename) on close(), so an interrupted run never leaves a truncated
// file that a resume (which skips existing sidecars) would silently keep.
class TargetWriter {
 public:
  TargetWriter(const std::string& path, uint32_t record_floats, const std::string& model_hash,
               uint32_t flags = 0);
  ~TargetWriter();  // closes if close() was not called

  TargetWriter(const TargetWriter&) = delete;
  TargetWriter& operator=(const TargetWriter&) = delete;

  // Append one position. `targets` is candidates.size() x record_floats,
  // candidate-major.
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<float>& targets);

  // Patch the header with the final position count and write the file.
  void close();

 private:
  std::string path_;
  uint32_t record_floats_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Loads a .mset file into memory and serves per-position views. Throws
// std::runtime_error on a missing file, bad magic, or version mismatch, so a
// stale file fails loudly rather than misparses.
class TargetReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  // Records are (Move, record_floats() floats) pairs, candidate-major, at
  // `records`; move_at/targets_at compute the offsets.
  struct Position {
    const TargetPositionHeader* header;
    const char* records;
  };

  explicit TargetReader(const std::string& path);

  uint32_t record_floats() const { return header_.record_floats; }
  uint32_t flags() const { return header_.flags; }
  std::string model_hash() const;
  int num_positions() const { return static_cast<int>(positions_.size()); }
  Position position(int i) const { return positions_[i]; }

  Move move_at(const Position& p, int candidate) const;
  const float* targets_at(const Position& p, int candidate) const;

 private:
  size_t record_bytes() const { return sizeof(Move) + sizeof(float) * header_.record_floats; }

  TargetFileHeader header_{};
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace move_set_eval
}  // namespace scribblez
