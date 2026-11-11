#include "data/group_log.h"

#include "data/sidecar_io.h"
#include "util/assert.h"
#include "util/exception.h"

#include <fstream>

namespace scribblez {

namespace {

GroupFileHeader make_header(const PoolSources& s, int map_sims, int map_steps) {
  GroupFileHeader h{};
  h.magic = kGroupMagic;
  h.version = kGroupFormatVersion;
  h.flags = s.leave_model.hash.empty() ? 0 : kPoolFlagLeaveInference;
  h.rollouts = uint32_t(s.rollouts);
  h.map_sims = uint32_t(map_sims);
  h.map_steps = uint32_t(map_steps);
  h.leaf_model = s.leaf_model.record("leaf model");
  h.leave_model = s.leave_model.record("leave model");
  return h;
}

GroupRack group_rack(const Rack& r) {
  GroupRack out{};
  for (int t = 0; t < TILE_KINDS; ++t) out.counts[size_t(t)] = uint8_t(r.count(Tile::of(t)));
  return out;
}

}  // namespace

std::vector<int> map_prefixes(int map_sims, int map_steps) {
  std::vector<int> out;
  for (int j = 1; j <= map_steps; ++j) out.push_back(map_sims * j / map_steps);
  return out;
}

GroupWriter::GroupWriter(const std::string& path, const PoolSources& sources, int map_sims,
                         int map_steps)
    : path_(path), map_sims_(size_t(map_sims)) {
  const GroupFileHeader h = make_header(sources, map_sims, map_steps);
  append_bytes(&buffer_, &h, sizeof(h));
}

GroupWriter::~GroupWriter() {
  if (!closed_ && std::uncaught_exceptions() == uncaught_at_open_) close();
}

void GroupWriter::add_group(const GroupRecord& g) {
  RELEASE_ASSERT(!closed_);
  RELEASE_ASSERT(g.pools.size() == g.moves.size() * g.racks.size());
  RELEASE_ASSERT(g.equity_ranks.size() == g.moves.size());
  const GroupHeader h{g.at.game_idx,
                      g.at.turn_idx,
                      uint32_t(g.moves.size()),
                      uint32_t(g.racks.size()),
                      uint32_t(g.maps.size()),
                      0};
  append_bytes(&buffer_, &h, sizeof(h));
  append_bytes(&buffer_, g.moves.data(), g.moves.size() * sizeof(Move));
  append_bytes(&buffer_, g.equity_ranks.data(), g.equity_ranks.size() * sizeof(int32_t));
  for (const Rack& r : g.racks) {
    const GroupRack gr = group_rack(r);
    append_bytes(&buffer_, &gr, sizeof(gr));
  }
  for (const PoolPosition& p : g.pools) append_pool_block(&buffer_, p, map_sims_);
  for (const GroupMap& m : g.maps) {
    const GroupMapHeader mh{uint16_t(m.move), uint16_t(m.rack), uint32_t(m.reply),
                            uint32_t(m.sims)};
    append_bytes(&buffer_, &mh, sizeof(mh));
    append_bytes(&buffer_, &m.maps, sizeof(m.maps));
  }
  ++num_groups_;
}

void GroupWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  reinterpret_cast<GroupFileHeader*>(buffer_.data())->num_groups = num_groups_;
  write_atomically(path_, buffer_);
}

GroupReader::GroupReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("GroupReader: cannot open {}", path);
  buffer_.resize(size_t(f.tellg()));
  f.seekg(0);
  f.read(buffer_.data(), std::streamsize(buffer_.size()));
  const std::string context = "GroupReader: " + path;
  size_t off = 0;
  take_bytes(buffer_, &off, sizeof(GroupFileHeader), context);
  if (header().magic != kGroupMagic) throw util::Exception("{}: bad magic", context);
  if (header().version != kGroupFormatVersion) {
    throw util::Exception("{}: format version {}, but this build reads {}", context,
                          header().version, kGroupFormatVersion);
  }
  for (uint32_t i = 0; i < header().num_groups; ++i) {
    Group g{};
    g.header =
      reinterpret_cast<const GroupHeader*>(take_bytes(buffer_, &off, sizeof(GroupHeader), context));
    g.moves = reinterpret_cast<const Move*>(
      take_bytes(buffer_, &off, g.header->num_moves * sizeof(Move), context));
    g.equity_ranks = reinterpret_cast<const int32_t*>(
      take_bytes(buffer_, &off, g.header->num_moves * sizeof(int32_t), context));
    g.racks = reinterpret_cast<const GroupRack*>(
      take_bytes(buffer_, &off, g.header->num_racks * sizeof(GroupRack), context));
    for (uint32_t p = 0; p < g.header->num_moves * g.header->num_racks; ++p)
      g.pools.push_back(take_pool_block(buffer_, &off, header().map_sims, context));
    for (uint32_t m = 0; m < g.header->num_maps; ++m) {
      g.maps.push_back(reinterpret_cast<const GroupMapHeader*>(
        take_bytes(buffer_, &off, sizeof(GroupMapHeader) + sizeof(rollout_abi::SimMaps), context)));
    }
    groups_.push_back(std::move(g));
  }
}

}  // namespace scribblez
