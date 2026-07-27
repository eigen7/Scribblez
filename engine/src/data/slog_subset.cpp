#include "data/slog_subset.h"

#include "data/binary_log.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <unordered_map>

namespace scribblez {
namespace binlog {

namespace {

// False on I/O failure.
bool read_file(const std::string& path, std::vector<char>& buf) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return true;
}

// Initial racks plus turn array.
int64_t blob_size(const GameMetadata& gm) {
  return static_cast<int64_t>(sizeof(InitialRacks)) +
         static_cast<int64_t>(gm.num_turns) * static_cast<int64_t>(sizeof(TurnBlob));
}

}  // namespace

bool write_slog_subset(const std::string& dst_path, const std::vector<SlogPick>& picks) {
  const uint32_t n = static_cast<uint32_t>(picks.size());

  // Cache source file contents so a file referenced by many picks is read once.
  std::unordered_map<std::string, std::vector<char>> cache;

  std::vector<GameMetadata> out_meta(n);
  std::vector<char> out_blobs;
  const int64_t blobs_start =
    static_cast<int64_t>(sizeof(FileHeader)) + static_cast<int64_t>(n) * sizeof(GameMetadata);

  for (uint32_t k = 0; k < n; ++k) {
    const SlogPick& pick = picks[k];
    auto it = cache.find(pick.path);
    if (it == cache.end()) {
      std::vector<char> buf;
      if (!read_file(pick.path, buf) || buf.size() < sizeof(FileHeader)) return false;
      const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
      if (hdr->magic != kMagic || hdr->version != kVersion) return false;
      it = cache.emplace(pick.path, std::move(buf)).first;
    }
    const std::vector<char>& src = it->second;
    const FileHeader* hdr = reinterpret_cast<const FileHeader*>(src.data());
    if (pick.game_idx < 0 || pick.game_idx >= static_cast<int64_t>(hdr->num_games)) return false;

    const GameMetadata* metas =
      reinterpret_cast<const GameMetadata*>(src.data() + sizeof(FileHeader));
    const GameMetadata& gm = metas[pick.game_idx];
    const int64_t bsize = blob_size(gm);
    if (gm.start_offset + static_cast<uint64_t>(bsize) > src.size()) return false;

    // Copy the metadata, repointing start_offset into the destination layout.
    out_meta[k] = gm;
    out_meta[k].start_offset = static_cast<uint64_t>(blobs_start) + out_blobs.size();
    out_blobs.insert(out_blobs.end(), src.data() + gm.start_offset,
                     src.data() + gm.start_offset + bsize);
  }

  // The subset's training-row count is the sum of the picked games' eligible
  // regions (eligible_begin/eligible_end are carried over per game above), so
  // the DataLoader sizes an epoch over the subset correctly.
  uint32_t num_sample_positions = 0;
  for (const GameMetadata& gm : out_meta)
    num_sample_positions += gm.eligible_end - gm.eligible_begin;

  FileHeader hdr{};
  hdr.magic = kMagic;
  hdr.version = kVersion;
  hdr.num_games = n;
  hdr.num_sample_positions = num_sample_positions;

  std::ofstream out(dst_path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char*>(out_meta.data()),
            static_cast<std::streamsize>(out_meta.size() * sizeof(GameMetadata)));
  out.write(out_blobs.data(), static_cast<std::streamsize>(out_blobs.size()));
  return static_cast<bool>(out);
}

}  // namespace binlog
}  // namespace scribblez
