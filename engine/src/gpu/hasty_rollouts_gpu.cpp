#include "gpu/hasty_rollouts_gpu.h"

#include "game/board.h"
#include "game/rack.h"
#include "game/tile.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_values.h"
#include "lexicon/word_map.h"
#include "rollout_kernel.h"

#include <bit>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

using rollout_gpu::HostTables;

// The leave's key: its tiles ascending, five bits each, 31 past the end.
uint32_t leave_key(const Rack& leave) {
  uint32_t key = 0;
  for (int i = 0; i < 6; ++i) {
    key |= uint32_t(i < leave.size() ? leave.tiles()[size_t(i)].index() : 31) << (5 * i);
  }
  return key;
}

void add_word_map(const WordMap& wm, HostTables& h) {
  for (int len = 2; len <= rollout_gpu::kMaxWordLen; ++len) {
    const WordMap::PerLength& pl = wm.per_length(len);
    h.slot_offset[len] = uint32_t(h.slots.size() / 3);
    if (pl.slots.empty()) {
      h.slots.insert(h.slots.end(), 3, 0);
      h.slot_mask[len] = 0;
    } else {
      h.slot_mask[len] = pl.mask;
      for (const WordMap::Slot& s : pl.slots) {
        h.slots.push_back(s.key.lo);
        h.slots.push_back(s.key.hi);
        h.slots.push_back(uint64_t(s.word_start) | (uint64_t(s.count) << 32));
      }
    }
    h.word_base[len] = uint32_t(h.words16.size() / 16);
    for (size_t i = 0; i < pl.words.size(); i += size_t(len)) {
      for (int j = 0; j < 16; ++j) {
        h.words16.push_back(j < len ? uint8_t(pl.words[i + size_t(j)].index()) : 0);
      }
    }
  }
}

struct BitRackHash {
  size_t operator()(const BitRack& k) const { return k.hash(); }
};
using BlankGroups = std::unordered_map<BitRack, std::vector<uint32_t>, BitRackHash>;

// Flattens one length's groups into the blank table `b`; each group's
// entries are 4-word packed words, listed back to back.
void add_blank_groups(const BlankGroups& groups, int len, int b, HostTables& h) {
  std::vector<uint64_t>& slots = h.blank_slots[b];
  h.blank_slot_offset[b][len] = uint32_t(slots.size() / 3);
  if (groups.empty()) {
    slots.insert(slots.end(), 3, 0);
    h.blank_slot_mask[b][len] = 0;
    return;
  }
  const uint64_t size = std::bit_ceil(groups.size() * 2);
  const uint64_t mask = size - 1;
  h.blank_slot_mask[b][len] = mask;
  const size_t base = slots.size();
  slots.insert(slots.end(), size * 3, 0);
  std::vector<uint32_t>& lists = h.blank_lists[b];
  for (const auto& [key, list] : groups) {
    uint64_t i = key.hash() & mask;
    while (slots[base + i * 3 + 2] != 0) i = (i + 1) & mask;
    slots[base + i * 3] = key.lo;
    slots[base + i * 3 + 1] = key.hi;
    slots[base + i * 3 + 2] = uint64_t(lists.size() / 4) | (uint64_t(list.size() / 4) << 32);
    lists.insert(lists.end(), list.begin(), list.end());
  }
}

// Every word of each length, under its letter multiset minus one letter (b =
// 0) and minus two (b = 1): the keys a one- and two-blank subrack probes. An
// entry carries the word and the removed letters, the blanks' designations.
void add_blank_tables(const WordMap& wm, HostTables& h) {
  for (int len = 2; len <= rollout_gpu::kMaxWordLen; ++len) {
    BlankGroups one, two;
    const WordMap::PerLength& pl = wm.per_length(len);
    std::vector<uint8_t> letters(static_cast<size_t>(len));
    uint32_t packed[4];
    for (const WordMap::Slot& s : pl.slots) {
      if (s.count == 0) continue;
      for (int l1 = 0; l1 < 26; ++l1) {
        if (s.key.get(l1) == 0) continue;
        BitRack k1 = s.key;
        k1.add_letter(l1, -1);
        std::vector<uint32_t>& g1 = one[k1];
        for (uint32_t w = s.word_start; w < s.word_start + s.count; ++w) {
          for (int i = 0; i < len; ++i) {
            letters[size_t(i)] = uint8_t(pl.words[size_t(w) * size_t(len) + size_t(i)].index());
          }
          rollout_gpu::pack_blank_entry(letters.data(), len, l1, 31, packed);
          g1.insert(g1.end(), packed, packed + 4);
        }
        for (int l2 = l1; l2 < 26; ++l2) {
          if (k1.get(l2) == 0) continue;
          BitRack k2 = k1;
          k2.add_letter(l2, -1);
          std::vector<uint32_t>& g2 = two[k2];
          for (uint32_t w = s.word_start; w < s.word_start + s.count; ++w) {
            for (int i = 0; i < len; ++i) {
              letters[size_t(i)] = uint8_t(pl.words[size_t(w) * size_t(len) + size_t(i)].index());
            }
            rollout_gpu::pack_blank_entry(letters.data(), len, l1, l2, packed);
            g2.insert(g2.end(), packed, packed + 4);
          }
        }
      }
    }
    add_blank_groups(one, len, 0, h);
    add_blank_groups(two, len, 1, h);
  }
}

void add_leaves(const LeaveValues& lv, HostTables& h) {
  std::vector<std::pair<uint32_t, float>> entries;
  lv.for_each(
    [&](const Rack& leave, float value) { entries.emplace_back(leave_key(leave), value); });
  const size_t size = std::bit_ceil(entries.size() * 2);
  h.leave_shift = 32 - std::countr_zero(size);
  h.leave_keys.assign(size, 0xFFFFFFFFu);
  h.leave_values.assign(size, 0.0f);
  for (const auto& [key, value] : entries) {
    uint32_t i = (key * 0x9E3779B1u) >> h.leave_shift;
    while (h.leave_keys[i] != 0xFFFFFFFFu) i = (i + 1) & uint32_t(size - 1);
    h.leave_keys[i] = key;
    h.leave_values[i] = value;
  }
}

HostTables build_tables(const Dictionary& dict, const HastyEquity& eq) {
  HostTables h;
  add_word_map(dict.word_map(), h);
  add_blank_tables(dict.word_map(), h);
  add_leaves(eq.leave_values(), h);
  h.arcs = dict.arcs();
  h.dawg_root = dict.root();
  for (int t = 0; t < rollout_gpu::kPegTiles; ++t) {
    for (int b = 0; b < rollout_gpu::kPegBags; ++b) h.peg[t][b] = eq.peg_for_tiles(t, b);
  }
  for (int i = 0; i < rollout_abi::kSquares; ++i) {
    h.letter_mult[i] = uint8_t(Board::PREMIUM[size_t(i)].letter_mult());
    h.word_mult[i] = uint8_t(Board::PREMIUM[size_t(i)].word_mult());
  }
  for (int L = 0; L < 26; ++L) h.tile_values[L] = TILE_VALUES[size_t(L)];
  for (int t = 0; t < TILE_KINDS; ++t) h.tile_distribution[t] = TILE_COUNTS[size_t(t)];
  return h;
}

}  // namespace

struct GpuHastyRollouts::Impl {
  rollout_gpu::DeviceTables* tables = nullptr;
};

GpuHastyRollouts::GpuHastyRollouts(const Dictionary& dict, const HastyEquity& eq)
    : impl_(std::make_unique<Impl>()) {
  const HostTables host = build_tables(dict, eq);
  const auto t0 = std::chrono::steady_clock::now();
  impl_->tables = rollout_gpu::upload_tables(host, &table_bytes_);
  table_upload_seconds_ =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

GpuHastyRollouts::~GpuHastyRollouts() { rollout_gpu::free_tables(impl_->tables); }

std::vector<rollout_abi::RolloutResult> GpuHastyRollouts::run(const HastyRolloutPosition& pos,
                                                              const Dictionary& dict, int n) {
  const rollout_abi::PositionAbi abi = to_abi(pos, dict);
  std::vector<rollout_abi::RolloutResult> results(static_cast<size_t>(n));
  double seconds[3];
  phase_cycles_.assign(rollout_gpu::kPhases, 0);
  counts_.assign(rollout_gpu::kCounts, 0);
  rollout_gpu::run_rollouts(impl_->tables, abi, n, results.data(), seconds, phase_cycles_.data(),
                            counts_.data());
  timings_ = Timings{seconds[0], seconds[1], seconds[2]};
  return results;
}

std::vector<GpuHastyRollouts::PhaseShare> GpuHastyRollouts::last_phase_shares() const {
  double total = 0;
  for (const unsigned long long c : phase_cycles_) total += double(c);
  std::vector<PhaseShare> out;
  for (int p = 0; p < rollout_gpu::kPhases; ++p) {
    out.push_back(
      {rollout_gpu::kPhaseNames[p], total > 0 ? double(phase_cycles_[size_t(p)]) / total : 0.0});
  }
  return out;
}

const rollout_abi::LeafState* GpuHastyRollouts::device_leaves() const {
  return rollout_gpu::device_leaves(impl_->tables);
}

void GpuHastyRollouts::encode_leaves(int first, int count, float* spatial, float* scalar) {
  rollout_gpu::encode_leaves(impl_->tables, first, count, spatial, scalar);
}

void GpuHastyRollouts::encode_leaves_to_host(int first, int count, float* spatial, float* scalar) {
  rollout_gpu::encode_leaves_to_host(impl_->tables, first, count, spatial, scalar);
}

std::vector<GpuHastyRollouts::CountValue> GpuHastyRollouts::last_counts() const {
  std::vector<CountValue> out;
  for (int c = 0; c < rollout_gpu::kCounts; ++c) {
    out.push_back({rollout_gpu::kCountNames[c], counts_[size_t(c)]});
  }
  return out;
}

}  // namespace scribblez
