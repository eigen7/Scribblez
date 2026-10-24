#pragma once

// The boundary between the engine (C++26, g++) and the CUDA translation unit
// (nvcc): flat tables in, results out. Only rollout_abi.h crosses it.

#include "gpu/rollout_abi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace scribblez::rollout_gpu {

constexpr int kMaxWordLen = 15;
constexpr int kPegTiles = 8;   // tiles played 0..7
constexpr int kPegBags = 101;  // bag size 0..100

// The kernel's phases, for the per-phase cycle counters thread 0 keeps.
enum Phase : int {
  kPhaseInit = 0,  // the start copy and the deal (rollout_abi.h)
  kPhaseLeaves,    // rack preparation, leave pricing, exchanges, span enumeration
  kPhaseSpans,     // span bounds and the item scan, per pass
  kPhaseItems,     // word-map probes, validation, scoring
  kPhaseApply,     // the reduction and the apply
  kPhaseCross,     // cross-check recomputation
  kPhases
};
// Work counters the kernel sums over all rollouts.
enum Count : int {
  kCountPlies = 0,
  kCountSpans,
  kCountItems,  // probes
  kCountHits,   // probes that found words
  kCountWords,  // words tried at a span
  kCountPlays,  // plays scored
  kCounts
};
inline constexpr const char* kCountNames[kCounts] = {"plies", "spans", "probes",
                                                     "hits",  "words", "plays"};
inline constexpr const char* kPhaseNames[kPhases] = {
  "init+draw", "leaves+exchanges", "spans+scan", "items", "reduce+apply", "cross-checks"};

// A blank-table entry, 16 bytes: the word's letters packed 5 bits each from
// bit 0 (so a list is one sequential stream of whole words), then the
// letters its one or two blanks take (31 for none).
constexpr int kBlankEntryLetterBits = 5;
constexpr int kBlankEntryXBit = 75;
constexpr int kBlankEntryYBit = 80;
inline void pack_blank_entry(const uint8_t* letters, int len, int x, int y, uint32_t out[4]) {
  uint64_t lo = 0, hi = 0;
  for (int i = 0; i < len; ++i) {
    const int bit = kBlankEntryLetterBits * i;
    const uint64_t v = letters[i];
    if (bit + kBlankEntryLetterBits <= 64) {
      lo |= v << bit;
    } else if (bit >= 64) {
      hi |= v << (bit - 64);
    } else {
      lo |= v << bit;
      hi |= v >> (64 - bit);
    }
  }
  hi |= uint64_t(x) << (kBlankEntryXBit - 64);
  hi |= uint64_t(y) << (kBlankEntryYBit - 64);
  out[0] = uint32_t(lo);
  out[1] = uint32_t(lo >> 32);
  out[2] = uint32_t(hi);
  out[3] = uint32_t(hi >> 32);
}

// The lexicon and equity tables laid out for the device.
struct HostTables {
  // The word map: per word length, an open-addressing table of 3-word slots
  // (key lo, key hi, word_start | count << 32; count 0 marks empty) over the
  // flat word array. A length with no words gets one empty slot.
  std::vector<uint64_t> slots;
  uint32_t slot_offset[kMaxWordLen + 1] = {};  // in slots
  uint64_t slot_mask[kMaxWordLen + 1] = {};
  // Every word padded to 16 bytes (one vector load), the lengths back to back.
  std::vector<uint8_t> words16;
  uint32_t word_base[kMaxWordLen + 1] = {};  // index of a length's first word
  // The blank tables (MAGPIE's): per word length and blank count b in {1, 2},
  // an open-addressing table over the multiset of a word's letters minus b
  // of them, to the indices (into that length's words) of the words that
  // contain the multiset. A subrack with b blanks probes it once; each word
  // found says what its blanks are. Slots as the word map's, with
  // list_start | count << 32 into `blank_lists`.
  std::vector<uint64_t> blank_slots[2];
  uint32_t blank_slot_offset[2][kMaxWordLen + 1] = {};
  uint64_t blank_slot_mask[2][kMaxWordLen + 1] = {};
  std::vector<uint32_t> blank_lists[2];  // 4 words per pack_blank_entry()
  // The leave table: open addressing over 32-bit keys (six 5-bit tile codes
  // ascending, 31 past the end), 0xFFFFFFFF empty.
  std::vector<uint32_t> leave_keys;
  std::vector<float> leave_values;
  int leave_shift = 0;  // 32 - log2(table size)
  // The KWG arcs, for cross-checks.
  std::vector<uint32_t> arcs;
  uint32_t dawg_root = 0;
  // Equity terms.
  double peg[kPegTiles][kPegBags] = {};
  uint8_t letter_mult[rollout_abi::kSquares] = {};
  uint8_t word_mult[rollout_abi::kSquares] = {};
  int tile_values[26] = {};
  int tile_distribution[rollout_abi::kTileKinds] = {};
};

// The leaf eval row as the device encoder writes it:
// encoding/leaf_eval_input_encoder.h's layout, which hasty_rollouts_gpu.cpp
// asserts these match.
constexpr int kLeafSpatialPlanes = 95;
constexpr int kLeafBlankPlane = 26;
constexpr int kLeafPremiumPlane0 = 27;
constexpr int kLeafCrossPlane0 = 31;
constexpr int kLeafJoinPlane0 = 83;
constexpr int kLeafScalars = 85;
constexpr int kLeafScoreDiff = 3 * rollout_abi::kTileKinds;  // after the three seats' counts
constexpr float kLeafScoreScale = 100.0f;
constexpr float kLeafFaceValueScale = 10.0f;
constexpr float kLeafLeaveValueScale = 25.0f;
constexpr float kLeafCountScale = 100.0f;

struct DeviceTables;

// Both throw std::runtime_error on a CUDA failure.
DeviceTables* upload_tables(const HostTables& host, size_t* bytes);
void free_tables(DeviceTables* tables);

// A batch in the device's layouts (sim/hasty_rollout_task.h's RolloutBatch).
struct BatchAbi {
  std::vector<rollout_abi::RolloutTask> tasks;
  std::vector<rollout_abi::PositionAbi> positions;
  std::vector<rollout_abi::GroupAbi> groups;
  std::vector<rollout_abi::OppLeaveAbi> opp_leaves;
};

// The batch's tasks into results[0, tasks), one block each. `seconds`
// receives the batch upload, the kernel and the result download;
// `phase_cycles` the summed per-block cycles of each Phase.
void run_rollouts(DeviceTables* tables, const BatchAbi& batch, rollout_abi::RolloutResult* results,
                  double seconds[3], unsigned long long phase_cycles[kPhases],
                  unsigned long long counts[kCounts]);

// The last run's leaf states, a device pointer (one per task).
const rollout_abi::LeafState* device_leaves(const DeviceTables* tables);

// LeafEvalInputEncoder rows for leaves [first, first + count) of the last
// run, written to device buffers: `spatial` (count x kLeafSpatialPlanes x
// 225 floats) and `scalar` (count x kLeafScalars). Synchronous.
void encode_leaves(DeviceTables* tables, int first, int count, float* spatial, float* scalar);

// encode_leaves into device scratch, copied back to host buffers of the
// same sizes.
void encode_leaves_to_host(DeviceTables* tables, int first, int count, float* spatial,
                           float* scalar);

}  // namespace scribblez::rollout_gpu
