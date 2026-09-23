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
//        record_planes x float                     plane scales
//        record_planes x kPlaneWidth x uint8       quantized planes]
//
// A position is identified by (game_index, turn_index) within the companion
// .slog file, addressing the PRE-move decision point; each record's targets
// describe the post-move state its Move produces.
//
// Placement planes
// ----------------
// A record's planes are the teacher's four placement heads at the candidate's
// post-move state, in kPlacementHeads order (opp_next, self_next, opp_win,
// self_win; also the SimObservation order). Each is a distribution over the
// kFootprintClasses footprint classes (training/footprint.h): the teacher's
// softmax masked by board legality only, illegal footprints at zero
// (masked_placement_distributions with no availability counts).
//
// Each plane is absmax-quantized to one byte per class: scale = max/255,
// byte = round(v/scale), value = byte * scale. The worst-case error of max/510
// per plane is ample for distillation targets. The planes stay dense because
// the masked footprint softmax is broad (measured: the top 128 classes hold
// only ~0.8-0.9 of the mass), so a sparse top-k would drop real tail mass, and
// fixed-width records keep both readers' vectorized indexing. A plane is thus
// ~13x the size of a per-cell (15x15) one: the cost of distilling the full
// footprint distribution rather than its per-cell marginal.
//
// record_planes is 0 or kTargetPlanes for the whole file. Stratified (training)
// files carry planes. Full-sweep files carry none: they are evaluation-only,
// their metrics are value-based, and their positions run to thousands of
// candidates, so planes would bloat them for no reader.
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
// Quantized placement planes per candidate record, when the file carries them.
inline constexpr uint32_t kTargetPlanes = 4;
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

// Quantizes one plane of kPlaneWidth probabilities into `out` and returns the
// scale (see "Placement planes" above). An all-zero plane gets scale 0.
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
  TargetWriter(const std::string& path, uint32_t record_floats, uint32_t record_planes,
               const std::string& model_hash, uint32_t flags = 0);
  ~TargetWriter();  // closes if close() was not called

  TargetWriter(const TargetWriter&) = delete;
  TargetWriter& operator=(const TargetWriter&) = delete;

  // `targets` is candidates.size() x record_floats and `planes` is
  // candidates.size() x record_planes x kPlaneWidth (empty for a plane-less
  // file), both candidate-major; the writer quantizes the planes.
  // `num_legal_moves` as in TargetPositionHeader.
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
  // record_planes() scales, then record_planes() x kPlaneWidth quantized
  // bytes; zero-length on a plane-less file.
  const float* plane_scales_at(const Position& p, int candidate) const;
  const uint8_t* planes_at(const Position& p, int candidate) const;

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
