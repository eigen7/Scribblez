#include "data/slog_sampling.h"

#include "util/exception.h"
#include "util/math.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace scribblez {
namespace binlog {

void sample_eligible_turns(const GameMetadata& gm, uint32_t game_idx, uint64_t run_seed,
                           int positions_per_game, std::vector<GamePositionIndex>* out) {
  const int begin = gm.eligible_begin;
  const int end = gm.eligible_end;
  if (begin >= end) return;
  if (positions_per_game <= 0) {
    for (int t = begin; t < end; ++t) out->push_back({game_idx, uint32_t(t)});
    return;
  }
  std::vector<uint32_t> turns(size_t(end - begin));
  std::iota(turns.begin(), turns.end(), uint32_t(begin));
  std::mt19937_64 rng(util::splitmix64(run_seed ^ util::splitmix64(0xC0FFEEull + game_idx)));
  std::shuffle(turns.begin(), turns.end(), rng);
  const int take = std::min<int>(positions_per_game, int(turns.size()));
  for (int i = 0; i < take; ++i) out->push_back({game_idx, turns[i]});
}

int count_eligible_sample(const GameMetadata& gm, int positions_per_game) {
  const int eligible = std::max(0, gm.eligible_end - gm.eligible_begin);
  return positions_per_game <= 0 ? eligible : std::min(positions_per_game, eligible);
}

uint64_t count_sampled_positions(const std::vector<char>& buf, int positions_per_game,
                                 int limit_games) {
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  const GameMetadata* metas =
    reinterpret_cast<const GameMetadata*>(buf.data() + sizeof(FileHeader));
  uint32_t num_games = hdr->num_games;
  if (limit_games > 0) num_games = std::min<uint32_t>(num_games, limit_games);
  uint64_t total = 0;
  for (uint32_t g = 0; g < num_games; ++g)
    total += uint64_t(count_eligible_sample(metas[g], positions_per_game));
  return total;
}

std::vector<char> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::CleanException("cannot open {}", path.string());
  const std::streamsize size = f.tellg();
  f.seekg(0);
  std::vector<char> bytes(size);
  f.read(bytes.data(), size);
  return bytes;
}

std::vector<std::filesystem::path> resolve_slog_inputs(const std::string& slog_dir,
                                                       const std::vector<std::string>& slog_files) {
  std::vector<std::filesystem::path> slogs;
  if (!slog_files.empty()) {
    for (const std::string& f : slog_files) slogs.emplace_back(f);
  } else if (!slog_dir.empty()) {
    for (const auto& entry : std::filesystem::directory_iterator(slog_dir))
      if (entry.path().extension() == ".slog") slogs.push_back(entry.path());
    std::sort(slogs.begin(), slogs.end());
  } else {
    throw util::CleanException("pass --slog-dir or --slog-file");
  }
  if (slogs.empty()) throw util::CleanException("no .slog files to process");
  return slogs;
}

std::vector<PendingSlog> load_pending_slogs(const std::vector<std::filesystem::path>& slogs,
                                            const char* sidecar_ext, bool accept_face_up,
                                            const char* face_up_error) {
  std::vector<PendingSlog> pending;
  for (const std::filesystem::path& slog : slogs) {
    std::filesystem::path sidecar = slog;
    sidecar.replace_extension(sidecar_ext);
    if (std::filesystem::exists(sidecar)) continue;
    std::vector<char> buf = read_file_bytes(slog);
    const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
    if (buf.size() < sizeof(FileHeader) || hdr->magic != kMagic || hdr->version != kVersion) {
      std::cerr << "  skip (bad header): " << slog.filename().string() << "\n";
      continue;
    }
    if ((hdr->flags & kFlagFaceUpLeaves) != 0 && !accept_face_up) {
      const std::string name = slog.filename().string();
      throw util::CleanException("{}", std::vformat(face_up_error, std::make_format_args(name)));
    }
    pending.push_back({slog, std::move(buf)});
  }
  return pending;
}

}  // namespace binlog
}  // namespace scribblez
