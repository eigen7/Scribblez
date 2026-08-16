#pragma once

// Binary sidecar format for move set evaluation model distillation targets.
// One .mset accompanies one .slog and holds, for a subset of its positions, the
// candidate moves sampled there and the teacher position evaluation model's
// readouts for each candidate's post-move state. The move set trainer pairs
// these with inputs it reconstructs by replay (docs/roadmap.md, track A).
//
// The header pins the teacher by content hash, so a corpus can be verified to
// come from one blessed checkpoint. It also carries the per-record target
// width, which makes the record layout head-extensible: readers consume the
// widths they know, and a future version can append heads without format
// surgery.
//
// File layout
// -----------
//   [TargetFileHeader                              88 B]
//   For each position p in [0, num_positions):
//     [TargetPositionHeader                        16 B]
//     [num_candidates(p) records, each:
//        Move                                      16 B
//        record_floats x float                     value targets
//        record_planes x float                     plane scales
//        record_planes x kPlaneCells x uint8       quantized planes]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file, addressing the PRE-move decision point; each record's targets
// describe the post-move state its Move produces.
//
// Placement planes
// ----------------
// A record's planes are the teacher's four placement masks at the candidate's
// post-move state -- per-cell probabilities in mask-head order (opp_next,
// self_next, opp_win, self_win; the SimObservation plane order). Each plane is
// absmax-quantized to a byte per cell: scale = max/255, cell = round(v/scale),
// value = cell * scale -- worst-case error max/510 per plane, ample for
// distillation targets. The planes stay dense: they are sigmoid outputs,
// near-zero rather than zero, so a nonzero-mask encoding saves an unreliable
// amount while costing the fixed-size records both readers index by; residual
// zero runs are left for generic file compression to collect.
//
// record_planes is 0 or kTargetPlanes for the whole file. Stratified (training)
// files carry planes; full-sweep files carry none -- they are evaluation-only
// and their gate metrics are value-based, while their positions run to
// thousands of candidates, which planes would grow 26x for no reader.
//
// A file holds one kind of position throughout, declared by
// kTargetFlagFullSweep: either the stratified training sample or the full-sweep
// evaluation slice. The two never mix, because a swept position is held out by
// construction and a mixed file could not be routed at file granularity.

#include "game/move.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace scribblez {
namespace move_set_eval {

// "MSET" in little-endian (bytes 'M','S','E','T' on disk).
inline constexpr uint32_t kTargetMagic = 0x5445534Du;
inline constexpr uint16_t kTargetVersion = 2;
// The value-target floats per candidate record, mover POV, in record order --
// the order the generator's inference loop scatters them in.
inline constexpr std::array<const char*, 5> kTargetNamesV1 = {"p_win", "p_draw", "p_loss",
                                                              "sd_mean", "sd_std"};
inline constexpr uint32_t kTargetFloatsV1 = kTargetNamesV1.size();
// Quantized placement planes per candidate record, when the file carries them.
inline constexpr uint32_t kTargetPlanes = 4;
inline constexpr uint32_t kPlaneCells = BOARD_SIZE * BOARD_SIZE;

// TargetFileHeader::flags bits, mirroring the .sobs convention.
inline constexpr uint32_t kTargetFlagOpenLeaves = 2u;
// Every position in this file is a full sweep of its legal candidates (capped;
// see TargetPositionHeader::num_legal_moves) rather than a stratified sample.
// Such a file is evaluation-only -- the A3 gate metrics need the tail moves the
// stratified sample cannot see -- so the trainer assigns it to the held-out
// side and never trains on it.
inline constexpr uint32_t kTargetFlagFullSweep = 4u;

// FP16 teacher inference can overflow the score-diff std readout to +inf on
// near-terminal states: the exported graph's Softplus is evaluated naively in
// half precision, overflowing for logits above ~11 even though the true
// softplus value there is about the logit itself. The generator stores the
// clamped std so every record carries finite targets -- the WLD label, the
// target that matters, is intact, and the std head is extrapolating on such
// states anyway. The cap sits comfortably above genuine teacher readouts
// (< ~90 across the shakeout corpus).
inline constexpr float kSdStdCap = 128.0f;

inline float clamped_sd_std(float sd_std) { return std::min(sd_std, kSdStdCap); }

inline constexpr int kTargetModelHashChars = 64;

// The target flags for a .mset generated from a .slog whose FileHeader carries
// `slog_flags`: the information condition the labeled games were played under,
// which the trainer must match with the student's input arm.
uint32_t target_flags_from_slog(uint16_t slog_flags);

// Absmax-quantize one placement plane of kPlaneCells probabilities into
// `out`, returning the scale (see "Placement planes" above). An all-zero
// plane gets scale 0 and all-zero cells, which dequantizes back to zero.
float quantize_plane(const float* values, uint8_t* out);

inline float dequantized_plane_value(uint8_t cell, float scale) { return cell * scale; }

#pragma pack(push, 1)

struct TargetFileHeader {
  uint32_t magic;    // kTargetMagic
  uint16_t version;  // kTargetVersion
  uint16_t reserved;
  uint32_t num_positions;
  uint32_t record_floats;                  // target floats per candidate record
  uint32_t record_planes;                  // quantized planes per record: 0 or kTargetPlanes
  uint32_t flags;                          // kTargetFlag* bits
  char model_hash[kTargetModelHashChars];  // hex, NUL-padded
};
static_assert(sizeof(TargetFileHeader) == 88, "TargetFileHeader must be 88 bytes");

struct TargetPositionHeader {
  uint32_t game_index;      // game within the companion .slog file
  uint32_t turn_index;      // pre-move turn the candidates were sampled at
  uint32_t num_candidates;  // records that follow
  // Legal moves the position had, so a sweep truncated by the candidate cap is
  // visible as num_candidates < num_legal_moves rather than passing for a
  // complete sweep. 0 means "not recorded" -- every stratified position, whose
  // candidate count says nothing about the position's size.
  uint32_t num_legal_moves;
};
static_assert(sizeof(TargetPositionHeader) == 16, "TargetPositionHeader must be 16 bytes");

#pragma pack(pop)

// Accumulates positions and writes the .mset file atomically (temp + rename) on
// close, so an interrupted run leaves no truncated file for a resume -- which
// skips existing sidecars -- to silently keep.
class TargetWriter {
 public:
  TargetWriter(const std::string& path, uint32_t record_floats, uint32_t record_planes,
               const std::string& model_hash, uint32_t flags = 0);
  ~TargetWriter();  // closes if close() was not called

  TargetWriter(const TargetWriter&) = delete;
  TargetWriter& operator=(const TargetWriter&) = delete;

  // `targets` is candidates.size() x record_floats and `planes` is
  // candidates.size() x record_planes x kPlaneCells (empty when the writer
  // carries no planes), both candidate-major; the writer quantizes each plane.
  // `num_legal_moves` is the position's legal-move count for a swept position,
  // 0 for a stratified one (see TargetPositionHeader).
  void add_position(uint32_t game_index, uint32_t turn_index, const std::vector<Move>& candidates,
                    const std::vector<float>& targets, const std::vector<float>& planes,
                    uint32_t num_legal_moves = 0);

  void close();

 private:
  std::string path_;
  uint32_t record_floats_;
  uint32_t record_planes_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
};

// Loads a .mset file into memory and serves per-position views. Throws
// util::Exception on a missing file, bad magic, or version mismatch, so a
// stale file fails loudly rather than misparsing.
class TargetReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  // `records` holds candidate-major records at the layout the file comment
  // documents, whose offsets move_at/targets_at/plane_scales_at/planes_at
  // compute.
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
  // record_planes() scales / record_planes() x kPlaneCells quantized cells;
  // meaningless (zero-length) on a plane-less file.
  const float* plane_scales_at(const Position& p, int candidate) const;
  const uint8_t* planes_at(const Position& p, int candidate) const;

 private:
  size_t record_bytes() const {
    return sizeof(Move) + sizeof(float) * header_.record_floats +
           header_.record_planes * (sizeof(float) + kPlaneCells);
  }

  TargetFileHeader header_{};
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace move_set_eval
}  // namespace scribblez
