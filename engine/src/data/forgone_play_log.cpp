#include "data/forgone_play_log.h"

#include "data/sidecar_io.h"
#include "game/board.h"
#include "game/game_log.h"
#include "lexicon/dictionary.h"
#include "util/assert.h"
#include "util/exception.h"
#include "util/hash.h"
#include "util/work_pool.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace scribblez {

namespace {

using belief::ForgonePlays;

const binlog::GameMetadata* slog_metadata(const char* slog) {
  return reinterpret_cast<const binlog::GameMetadata*>(slog + sizeof(binlog::FileHeader));
}

uint32_t slog_num_games(const char* slog) {
  return reinterpret_cast<const binlog::FileHeader*>(slog)->num_games;
}

void copy_hash(char (&dst)[kForgoneHashSize], const std::string& hash) {
  std::memset(dst, 0, kForgoneHashSize);
  std::memcpy(dst, hash.data(), std::min(hash.size(), kForgoneHashSize));
}

bool hash_equals(const char (&field)[kForgoneHashSize], const std::string& hash) {
  char expected[kForgoneHashSize];
  copy_hash(expected, hash);
  return std::memcmp(field, expected, kForgoneHashSize) == 0;
}

void check_header(const std::string& path, const ForgoneFileHeader& h, uint32_t num_games,
                  const std::string& lexicon_hash, const std::string& slog_binding) {
  if (h.magic != kForgoneMagic) throw util::Exception("{}: not a .forgone file", path);
  if (h.version != kForgoneVersion) {
    throw util::Exception("{}: version {}, expected {}", path, h.version, kForgoneVersion);
  }
  if (h.num_games != num_games) {
    throw util::Exception("{}: {} games, its .slog has {}", path, h.num_games, num_games);
  }
  if (!hash_equals(h.lexicon_hash, lexicon_hash)) {
    throw util::Exception("{}: generated under another lexicon", path);
  }
  if (!hash_equals(h.slog_binding, slog_binding)) {
    throw util::Exception("{}: does not match its .slog (regenerated since?)", path);
  }
}

// Every game's entries.
std::vector<std::vector<ForgonePlays>> all_games(const std::vector<char>& slog,
                                                 const Dictionary& dict) {
  const uint32_t num_games = slog_num_games(slog.data());
  const binlog::GameMetadata* metas = slog_metadata(slog.data());
  std::vector<std::vector<ForgonePlays>> out(num_games);
  util::parallel_lanes(num_games, [&](util::LaneCursor& cursor) {
    std::vector<TurnRecord> scratch;
    for (size_t g; cursor.next(g);) {
      const GameLog log = binlog::make_game_view(slog.data(), uint32_t(g), scratch, nullptr);
      out[g] = forgone_plays_of_game(log, forgone_turns(metas[g]), dict);
    }
  });
  return out;
}

}  // namespace

std::string forgone_slog_binding(const binlog::GameMetadata* metas, uint32_t num_games) {
  std::vector<binlog::GameMetadata> copy(metas, metas + num_games);
  for (binlog::GameMetadata& m : copy) m.sampled_turn = 0;
  return util::content_hash(
    {reinterpret_cast<const char*>(copy.data()), copy.size() * sizeof(binlog::GameMetadata)});
}

std::string forgone_slog_binding(const char* slog) {
  return forgone_slog_binding(slog_metadata(slog), slog_num_games(slog));
}

std::vector<ForgonePlays> forgone_plays_of_game(const GameLog& g, int turns,
                                                const Dictionary& dict) {
  std::vector<ForgonePlays> out;
  out.reserve(size_t(turns));
  Board board;
  for (int k = 0; k < turns; ++k) {
    const Move& move = g.records[k].move;
    out.push_back(belief::compute_forgone_plays(board, dict, move));
    if (move.type() == MoveType::PLAY) board.apply(move);
  }
  return out;
}

void write_forgone_sidecar(const std::vector<char>& slog, const Dictionary& dict,
                           const std::string& path) {
  const std::vector<std::vector<ForgonePlays>> games = all_games(slog, dict);
  ForgoneFileHeader h{};
  h.magic = kForgoneMagic;
  h.version = kForgoneVersion;
  h.num_games = uint32_t(games.size());
  copy_hash(h.lexicon_hash, dict.content_hash());
  copy_hash(h.slog_binding, forgone_slog_binding(slog.data()));

  std::vector<char> bytes;
  append_bytes(&bytes, &h, sizeof(h));
  uint32_t first = 0;
  for (const std::vector<ForgonePlays>& entries : games) {
    append_bytes(&bytes, &first, sizeof(first));
    first += uint32_t(entries.size());
  }
  append_bytes(&bytes, &first, sizeof(first));
  for (const std::vector<ForgonePlays>& entries : games) {
    append_bytes(&bytes, entries.data(), entries.size() * sizeof(ForgonePlays));
  }
  write_atomically(path, bytes);
}

std::string forgone_path_of(const std::string& slog_path) {
  return std::filesystem::path(slog_path).replace_extension(kForgoneExtension).string();
}

void validate_forgone_file(const std::string& path, uint32_t num_games,
                           const std::string& lexicon_hash, const std::string& slog_binding) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw util::Exception("{}: missing (every .slog needs its .forgone sidecar)", path);
  ForgoneFileHeader h;
  if (!in.read(reinterpret_cast<char*>(&h), sizeof(h))) {
    throw util::Exception("{}: truncated header", path);
  }
  check_header(path, h, num_games, lexicon_hash, slog_binding);
}

ForgoneSidecar::ForgoneSidecar(const std::string& path, const char* slog,
                               const std::string& lexicon_hash) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw util::Exception("{}: missing (every .slog needs its .forgone sidecar)", path);
  bytes_.assign(std::istreambuf_iterator<char>(in), {});
  if (bytes_.size() < sizeof(ForgoneFileHeader)) {
    throw util::Exception("{}: truncated header", path);
  }
  const auto& h = *reinterpret_cast<const ForgoneFileHeader*>(bytes_.data());
  num_games_ = slog_num_games(slog);
  check_header(path, h, num_games_, lexicon_hash, forgone_slog_binding(slog));
  const size_t index_end = sizeof(h) + (size_t(num_games_) + 1) * sizeof(uint32_t);
  if (bytes_.size() < index_end) throw util::Exception("{}: truncated index", path);
  const uint32_t entries = reinterpret_cast<const uint32_t*>(bytes_.data() + sizeof(h))[num_games_];
  if (bytes_.size() != index_end + size_t(entries) * sizeof(ForgonePlays)) {
    throw util::Exception("{}: body does not match its index", path);
  }
}

std::span<const ForgonePlays> ForgoneSidecar::game(uint32_t g) const {
  RELEASE_ASSERT(g < num_games_);
  const uint32_t* first =
    reinterpret_cast<const uint32_t*>(bytes_.data() + sizeof(ForgoneFileHeader));
  const auto* entries = reinterpret_cast<const ForgonePlays*>(
    bytes_.data() + sizeof(ForgoneFileHeader) + (size_t(num_games_) + 1) * sizeof(uint32_t));
  return {entries + first[g], entries + first[g + 1]};
}

}  // namespace scribblez
