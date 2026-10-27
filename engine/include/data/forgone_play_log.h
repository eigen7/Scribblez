#pragma once

// The .forgone sidecar: the forgone plays (belief/forgone_plays.h) of every
// move a leave inference row of the companion .slog reads. One .forgone file
// accompanies one .slog, with the same stem. Computing the margins means a
// move generation per move, too slow to redo at every replay, so they are
// computed once, where the games are generated, and read back
// (docs/architecture.md, "The replay-reconstruction invariant").
//
// File layout
// -----------
//   [ForgoneFileHeader                          44 B]
//   [uint32_t first_entry[num_games + 1]            ]  game g's entries are
//                                                     [first_entry[g], first_entry[g + 1])
//   [ForgonePlays entries                   27 B each]
//
// Game g has one entry per turn k in [0, forgone_turns(metadata of g)), in
// turn order, indexed by the .slog's game index.

#include "belief/forgone_plays.h"
#include "data/binary_log.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;
struct GameLog;

// "FGON" in little-endian.
inline constexpr uint32_t kForgoneMagic = 0x4E4F4746u;
inline constexpr uint16_t kForgoneVersion = 1;
inline constexpr const char* kForgoneExtension = ".forgone";
inline constexpr size_t kForgoneHashSize = 16;  // util::content_hash's hex digits

static_assert(sizeof(belief::ForgonePlays) == TILE_KINDS);

#pragma pack(push, 1)

struct ForgoneFileHeader {
  uint32_t magic;    // kForgoneMagic
  uint16_t version;  // kForgoneVersion
  uint16_t reserved;
  uint32_t num_games;
  // Dictionary::content_hash of the lexicon the margins were generated under.
  char lexicon_hash[kForgoneHashSize];
  // forgone_slog_binding of the companion .slog, so a regenerated .slog
  // cannot pass with a stale sidecar.
  char slog_binding[kForgoneHashSize];
};
static_assert(sizeof(ForgoneFileHeader) == 44, "ForgoneFileHeader must be 44 bytes");

#pragma pack(pop)

// The turns of a game that have an entry. A leave inference row at turn
// t < eligible_end reads the moves at t - 1, t - 3 and t - 5.
inline int forgone_turns(const binlog::GameMetadata& gm) {
  return std::max(0, gm.eligible_end - 1);
}

// A fingerprint of a .slog's game metadata table, without sampled_turn (a
// random draw that differs between otherwise identical files).
std::string forgone_slog_binding(const binlog::GameMetadata* metas, uint32_t num_games);
// The same, of a loaded .slog.
std::string forgone_slog_binding(const char* slog);

// The first `turns` entries of game `g`, by replay.
std::vector<belief::ForgonePlays> forgone_plays_of_game(const GameLog& g, int turns,
                                                        const Dictionary& dict);

// Computes the sidecar of the loaded, validated .slog `slog` and writes it
// atomically to `path`, its games in parallel.
void write_forgone_sidecar(const std::vector<char>& slog, const Dictionary& dict,
                           const std::string& path);

// The sidecar path of a .slog path.
std::string forgone_path_of(const std::string& slog_path);

// Reads only the header of the .forgone file at `path` and checks it against
// its .slog. Throws util::Exception naming `path` on a missing or truncated
// file, bad magic or version, or a game count, lexicon or binding that
// differs.
void validate_forgone_file(const std::string& path, uint32_t num_games,
                           const std::string& lexicon_hash, const std::string& slog_binding);

// A whole .forgone file in memory, checked against its .slog, serving each
// game's entries.
class ForgoneSidecar {
 public:
  // Throws util::Exception as validate_forgone_file does, or on a body that
  // does not match the header.
  ForgoneSidecar(const std::string& path, const char* slog, const std::string& lexicon_hash);

  // Game `g`'s entries, one per turn from 0.
  std::span<const belief::ForgonePlays> game(uint32_t g) const;

  size_t bytes() const { return bytes_.size(); }

 private:
  std::vector<char> bytes_;
  uint32_t num_games_ = 0;
};

}  // namespace scribblez
