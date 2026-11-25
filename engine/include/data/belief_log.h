#pragma once

// The .belief sidecar: at every turn of the companion .slog that a leave
// inference row is made from, the mover's belief about the opponent's kept
// leave under one leave inference model, with the summaries of it the leaf
// network reads in belief mode (docs/plans/deep_bot_belief_leaf.md). A
// summary takes a pass over every leave of the belief's length
// (LeaveSampler::moments), about 21 ms at length 6, too slow to redo per row
// per epoch, so the summaries are computed once and read back.
//
// File layout
// -----------
//   [BeliefFileHeader                          100 B]
//   [uint32_t first_entry[num_games + 1]            ]  game g's entries are
//                                                     [first_entry[g], first_entry[g + 1])
//   [BeliefEntry entries                    91 B each]
//   [draws × RACK_SIZE tile codes per entry          ]  when draws > 0
//
// Game g has one entry per turn of the leave inference rows' region,
// [eligible_begin, eligible_end) of its metadata, in turn order. A turn that
// begins with an empty bag has none, though the leaf's rows go on past it
// (BlockDecoder::row_turns): the opponent holds every unseen tile, so the
// belief is certain and a belief-mode row reads that rack as known mode does.
//
// Precision: the parameters and value moments are float16 (relative error up
// to 2^-11); an expected count is an 8-bit fraction of the leave's length
// (absolute error up to length / 510 tiles).

#include "belief/leave_params.h"
#include "data/binary_log.h"
#include "game/rack.h"
#include "game/tile.h"
#include "nn/onnx_metadata.h"
#include "training/leave_inference_task.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
class LeaveValueArrays;

// "BLIF" in little-endian.
inline constexpr uint32_t kBeliefMagic = 0x46494C42u;
inline constexpr uint16_t kBeliefVersion = 1;
inline constexpr const char* kBeliefExtension = ".belief";
inline constexpr size_t kBeliefHashSize = 16;  // util::content_hash's hex digits
inline constexpr size_t kBeliefTagSize = 32;

#pragma pack(push, 1)

struct BeliefFileHeader {
  uint32_t magic;    // kBeliefMagic
  uint16_t version;  // kBeliefVersion
  uint16_t draws;    // the leaves stored per entry; 0 outside the held-out set
  uint32_t num_games;
  // The leave inference model's identity (nn::ModelIdentity).
  char model_hash[kBeliefHashSize];
  char model_tag[kBeliefTagSize];
  int32_t model_generation;
  int32_t model_format_version;
  // LeaveValueArrays::content_hash of the table the summaries are over.
  char leave_values_hash[kBeliefHashSize];
  // forgone_slog_binding of the companion .slog.
  char slog_binding[kBeliefHashSize];
};
static_assert(sizeof(BeliefFileHeader) == 100, "BeliefFileHeader must be 100 bytes");

// One turn's belief, stored compactly (see the file comment's precision).
struct BeliefEntry {
  uint8_t cause;   // LeaveRowCause: kValid where the model has a belief
  uint8_t length;  // of the leave
  // float16 bits of belief::LeaveParams: theta by Tile code, then a and b;
  // zero unless kValid.
  std::array<uint16_t, belief::LeaveParams::kFloats> params;
  uint16_t value_mean;                       // float16 bits, points
  uint16_t value_sd;                         // float16 bits, points
  std::array<uint8_t, TILE_KINDS> expected;  // 255 × expected count / length, by Tile code
};
static_assert(sizeof(BeliefEntry) == 91, "BeliefEntry must be 91 bytes");

#pragma pack(pop)

// A BeliefEntry in full precision.
struct LeaveBelief {
  LeaveRowCause cause = LeaveRowCause::kValid;
  int length = 0;
  belief::LeaveParams params{};
  double value_mean = 0;
  double value_sd = 0;
  std::array<double, TILE_KINDS> expected{};  // the expected count of each kind in the leave

  BeliefEntry pack() const;
  static LeaveBelief unpack(const BeliefEntry& e);
};

// The sidecar path of a .slog path.
std::string belief_path_of(const std::string& slog_path);

// The leave inference model a sidecar's beliefs are under.
struct BeliefModel {
  belief::LeaveParamsFn predict;
  nn::ModelIdentity identity;
};

// The jitter of the leaves stored for game `game`'s turn `turn`, the seed
// belief::opp_leave_draws takes.
uint64_t belief_draw_seed(uint32_t game, int turn);

// Computes the sidecar of the loaded, validated .slog `slog` at `slog_path`,
// whose .forgone sidecar must be in place, and writes it atomically to
// `path`. With `draws` > 0 each entry also stores that many of the
// opponent's leaves, drawn as DeepBot draws them (belief/leave_draws.h) with
// belief_draw_seed's jitter.
void write_belief_sidecar(const std::vector<char>& slog, const std::string& slog_path,
                          const Dictionary& dict, const LeaveValueArrays& table,
                          const BeliefModel& model, int draws, const std::string& path);

// A whole .belief file in memory, checked against its .slog and leave table,
// serving each game's entries. Which model it is under is the caller's to
// check (identity()).
class BeliefSidecar {
 public:
  // Throws util::Exception naming `path` on a missing or truncated file, bad
  // magic or version, a game count, binding or leave table that differs, or a
  // body that does not match the header.
  BeliefSidecar(const std::string& path, const char* slog, const std::string& leave_values_hash);

  nn::ModelIdentity identity() const;
  int draws() const { return header().draws; }

  // Game `g`'s entries, one per turn of its region from eligible_begin.
  std::span<const BeliefEntry> game(uint32_t g) const;
  // The `i`-th stored leave of game `g`'s `k`-th entry; requires draws() > 0.
  Rack draw(uint32_t g, size_t k, int i) const;

  size_t bytes() const { return bytes_.size(); }

 private:
  const BeliefFileHeader& header() const;
  const uint32_t* first_entry() const;
  size_t entries_offset() const;

  std::vector<char> bytes_;
};

}  // namespace scribblez
