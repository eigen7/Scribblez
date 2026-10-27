// slog_migrate_v11: rewrites version-10 .slog files to version 11 in place,
// for the corpora written before GameMetadata gained the game's seed and seat
// players. Those fields read as unknown (seed 0, seats -1) in a migrated file;
// every other byte of a game is kept.
//
//   slog_migrate_v11 /workspace/mount/tags /workspace/mount/win_tables
//
// Each argument is searched recursively. A .forgone sidecar binds to its .slog
// by a hash of the metadata table, which the migration changes, so a sidecar
// that matched its v10 .slog is rebound to the v11 one. Files are replaced
// through a temp file and a rename, and a .slog already at v11 is skipped but
// its sidecar still checked, so an interrupted run is finished by rerunning it.
//
// One-shot: delete this tool once every machine's corpora are migrated.

#include "data/binary_log.h"
#include "data/forgone_play_log.h"
#include "data/sidecar_io.h"
#include "util/exception.h"
#include "util/hash.h"
#include "util/io.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace scribblez;
namespace fs = std::filesystem;

constexpr uint16_t kOldVersion = 10;
static_assert(binlog::kVersion == 11, "this tool migrates v10 to v11 only");

#pragma pack(push, 1)
// binlog::GameMetadata as version 10 laid it out.
struct GameMetadataV10 {
  uint64_t start_offset;
  uint32_t num_turns;
  uint16_t sampled_turn;
  int16_t final_score_p0;
  int16_t final_score_p1;
  int16_t initial_score_p0;
  int16_t initial_score_p1;
  uint8_t eligible_begin;
  uint8_t eligible_end;
};
#pragma pack(pop)
static_assert(sizeof(GameMetadataV10) == 24);

constexpr uint64_t kGrowthPerGame = sizeof(binlog::GameMetadata) - sizeof(GameMetadataV10);

struct Counts {
  int migrated = 0;
  int current = 0;
  int rebound = 0;
  int stale_sidecars = 0;
};

const binlog::FileHeader& header_of(const std::vector<char>& slog) {
  return *reinterpret_cast<const binlog::FileHeader*>(slog.data());
}

binlog::GameMetadata upgrade(const GameMetadataV10& old, uint32_t num_games) {
  binlog::GameMetadata gm{};
  gm.start_offset = old.start_offset + num_games * kGrowthPerGame;
  gm.num_turns = old.num_turns;
  gm.sampled_turn = old.sampled_turn;
  gm.final_score_p0 = old.final_score_p0;
  gm.final_score_p1 = old.final_score_p1;
  gm.initial_score_p0 = old.initial_score_p0;
  gm.initial_score_p1 = old.initial_score_p1;
  gm.eligible_begin = old.eligible_begin;
  gm.eligible_end = old.eligible_end;
  gm.seed = 0;
  gm.seat0_player = -1;
  gm.seat1_player = -1;
  return gm;
}

GameMetadataV10 downgrade(const binlog::GameMetadata& gm, uint32_t num_games) {
  return {gm.start_offset - num_games * kGrowthPerGame,
          gm.num_turns,
          gm.sampled_turn,
          gm.final_score_p0,
          gm.final_score_p1,
          gm.initial_score_p0,
          gm.initial_score_p1,
          gm.eligible_begin,
          gm.eligible_end};
}

// The v11 bytes of the v10 file `slog`: the header's version, the widened
// metadata table, and the games' bytes unchanged after it.
std::vector<char> migrated(const std::vector<char>& slog) {
  binlog::FileHeader header = header_of(slog);
  header.version = binlog::kVersion;
  const auto* old = reinterpret_cast<const GameMetadataV10*>(slog.data() + sizeof(header));
  std::vector<char> out;
  append_bytes(&out, &header, sizeof(header));
  for (uint32_t g = 0; g < header.num_games; ++g) {
    const binlog::GameMetadata gm = upgrade(old[g], header.num_games);
    append_bytes(&out, &gm, sizeof(gm));
  }
  const size_t body = sizeof(header) + header.num_games * sizeof(GameMetadataV10);
  append_bytes(&out, slog.data() + body, slog.size() - body);
  return out;
}

// forgone_slog_binding as version 10 computed it, of the v11 file `slog`.
std::string v10_binding(const std::vector<char>& slog) {
  const uint32_t n = header_of(slog).num_games;
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(slog.data() + sizeof(binlog::FileHeader));
  std::vector<GameMetadataV10> old;
  for (uint32_t g = 0; g < n; ++g) {
    old.push_back(downgrade(metas[g], n));
    old.back().sampled_turn = 0;
  }
  return util::content_hash(
    {reinterpret_cast<const char*>(old.data()), old.size() * sizeof(GameMetadataV10)});
}

bool binding_is(const ForgoneFileHeader& h, const std::string& hash) {
  char expected[kForgoneHashSize] = {};
  std::memcpy(expected, hash.data(), std::min(hash.size(), kForgoneHashSize));
  return std::memcmp(h.slog_binding, expected, kForgoneHashSize) == 0;
}

// Rebinds the sidecar of the v11 file `slog` at `slog_path` if it was bound to
// the file's v10 form.
void rebind_sidecar(const std::string& slog_path, const std::vector<char>& slog, Counts& counts) {
  const std::string path = forgone_path_of(slog_path);
  if (!fs::exists(path)) return;
  std::vector<char> sidecar = util::read_file_bytes(path);
  auto& h = *reinterpret_cast<ForgoneFileHeader*>(sidecar.data());
  const std::string binding = forgone_slog_binding(slog.data());
  if (binding_is(h, binding)) return;
  if (!binding_is(h, v10_binding(slog))) {
    std::cerr << path << ": bound to neither form of its .slog; left as is\n";
    ++counts.stale_sidecars;
    return;
  }
  std::memset(h.slog_binding, 0, kForgoneHashSize);
  std::memcpy(h.slog_binding, binding.data(), std::min(binding.size(), kForgoneHashSize));
  write_atomically(path, sidecar);
  ++counts.rebound;
}

void migrate_file(const fs::path& path, Counts& counts) {
  std::vector<char> slog = util::read_file_bytes(path.string());
  if (slog.size() < sizeof(binlog::FileHeader) || header_of(slog).magic != binlog::kMagic) {
    throw util::CleanException("{}: not a .slog", path.string());
  }
  const uint16_t version = header_of(slog).version;
  if (version == kOldVersion) {
    slog = migrated(slog);
    write_atomically(path.string(), slog);
    ++counts.migrated;
  } else if (version == binlog::kVersion) {
    ++counts.current;
  } else {
    throw util::CleanException("{}: version {}, expected {} or {}", path.string(), version,
                               kOldVersion, binlog::kVersion);
  }
  rebind_sidecar(path.string(), slog, counts);
}

void migrate_tree(const fs::path& root, Counts& counts) {
  for (const auto& entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file() && entry.path().extension() == ".slog") {
      migrate_file(entry.path(), counts);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: slog_migrate_v11 DIR [DIR...]\n";
    return 2;
  }
  try {
    Counts counts;
    for (int i = 1; i < argc; ++i) migrate_tree(argv[i], counts);
    std::cout << counts.migrated << " migrated, " << counts.current << " already v11, "
              << counts.rebound << " sidecars rebound, " << counts.stale_sidecars
              << " sidecars stale\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "slog_migrate_v11: " << e.what() << "\n";
    return 1;
  }
}
