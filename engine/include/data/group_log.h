#pragma once

// The .group sidecar: grouped pools, the search network's store input's
// training data (docs/plans/deep_bot.md, "Training data: grouped pools"). A
// companion of a .slog, whose (game_index, turn_index) names each group's
// parent position, with Alice to move. A group holds her proposer's moves
// there, the racks sampled for Bob, Bob's pool at each (move, rack) in the
// .pool record's form (data/pool_log.h), and the maps of each rack's leading
// replies, each summed over the first n of the reply's rollouts.
//
// File layout (little-endian, packed):
//
//   [GroupFileHeader                                       136 B]
//   per group:
//     [GroupHeader                                          24 B]
//     [Move x num_moves                   Alice's, best equity first]
//     [GroupRack x num_racks                                28 B each]
//     per (move, rack), move-major: a pool block (data/pool_log.h), each
//       candidate's values cut to its first map_sims rollouts
//     per map:
//       [GroupMapHeader                                     12 B]
//       [rollout_abi::SimMaps              float32 x sim_maps_fields()]

#include "data/pool_log.h"
#include "data/slog_sampling.h"
#include "game/move.h"
#include "game/rack.h"
#include "gpu/rollout_abi.h"

#include <array>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace scribblez {

// "GRUP" in little-endian.
inline constexpr uint32_t kGroupMagic = 0x50555247u;
// The format version (docs/plans/deep_bot_v0.md, "Versions"). Bump it
// whenever the layout or a field's meaning changes; readers accept only this
// version.
inline constexpr uint16_t kGroupFormatVersion = 1;
inline constexpr const char* kGroupExtension = ".group";

#pragma pack(push, 1)

struct GroupFileHeader {
  uint32_t magic;    // kGroupMagic
  uint16_t version;  // kGroupFormatVersion
  uint16_t flags;    // kPoolFlag* bits
  uint32_t num_groups;
  uint32_t rollouts;  // the pools' budget per reply; racing stops some early
  // The maps' rollout counts: a search's count per rack, and the steps up to
  // it, map_sims * j / map_steps for j in 1..map_steps.
  uint32_t map_sims;
  uint32_t map_steps;
  PoolModelRecord leaf_model;
  PoolModelRecord leave_model;
};
static_assert(sizeof(GroupFileHeader) == 136, "GroupFileHeader must be 136 bytes");

struct GroupHeader {
  uint32_t game_index;  // game within the companion .slog file
  uint32_t turn_index;  // the parent: the pre-move turn, Alice to move
  uint32_t num_moves;
  uint32_t num_racks;
  uint32_t num_maps;
  uint32_t reserved;
};
static_assert(sizeof(GroupHeader) == 24, "GroupHeader must be 24 bytes");

// A rack as tile counts, by Tile code.
struct GroupRack {
  std::array<uint8_t, TILE_KINDS> counts;
  uint8_t reserved;
};
static_assert(sizeof(GroupRack) == 28, "GroupRack must be 28 bytes");

struct GroupMapHeader {
  uint16_t move;   // index into the group's moves
  uint16_t rack;   // index into the group's racks
  uint32_t reply;  // index into that (move, rack) pool's candidates
  uint32_t sims;   // the rollouts summed: the reply's first this many
};
static_assert(sizeof(GroupMapHeader) == 12, "GroupMapHeader must be 12 bytes");

#pragma pack(pop)

// One reply's sims summed into its maps, as a writer takes it.
struct GroupMap {
  int move = 0;
  int rack = 0;
  int reply = 0;
  int sims = 0;
  rollout_abi::SimMaps maps;
};

// One group, as a writer takes it.
struct GroupRecord {
  binlog::GamePositionIndex at;
  std::vector<Move> moves;
  std::vector<Rack> racks;
  std::vector<PoolPosition> pools;  // move-major: pools[move * racks.size() + rack]
  std::vector<GroupMap> maps;
};

// The maps' rollout counts a file's header names, ascending.
std::vector<int> map_prefixes(int map_sims, int map_steps);

// Accumulates groups in memory and writes the .group file atomically on
// close().
class GroupWriter {
 public:
  GroupWriter(const std::string& path, const PoolSources& sources, int map_sims, int map_steps);
  // Closes if close() was not called, unless an exception is unwinding: a
  // partial file would pass for a finished one, so it is dropped instead.
  ~GroupWriter();

  GroupWriter(const GroupWriter&) = delete;
  GroupWriter& operator=(const GroupWriter&) = delete;

  void add_group(const GroupRecord& g);
  // The bytes written so far, the header included.
  size_t bytes() const { return buffer_.size(); }
  void close();

 private:
  std::string path_;
  size_t map_sims_;
  std::vector<char> buffer_;
  uint32_t num_groups_ = 0;
  bool closed_ = false;
  int uncaught_at_open_ = std::uncaught_exceptions();
};

// Loads a whole .group file and serves per-group views. Throws
// util::Exception on a missing or truncated file, bad magic, or a format
// version other than kGroupFormatVersion.
class GroupReader {
 public:
  // Per-group view into the reader's buffer; valid while the reader lives.
  struct Group {
    const GroupHeader* header;
    const Move* moves;                        // header->num_moves
    const GroupRack* racks;                   // header->num_racks
    std::vector<PoolPositionView> pools;      // move-major
    std::vector<const GroupMapHeader*> maps;  // header->num_maps, each followed by its SimMaps
  };

  explicit GroupReader(const std::string& path);

  const GroupFileHeader& header() const {
    return *reinterpret_cast<const GroupFileHeader*>(buffer_.data());
  }
  int num_groups() const { return int(groups_.size()); }
  const Group& group(int i) const { return groups_[size_t(i)]; }
  // The maps a map header is followed by.
  static const rollout_abi::SimMaps& maps_of(const GroupMapHeader* h) {
    return *reinterpret_cast<const rollout_abi::SimMaps*>(h + 1);
  }

 private:
  std::vector<char> buffer_;
  std::vector<Group> groups_;
};

}  // namespace scribblez
