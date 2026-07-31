#pragma once

// Binary game-log format for training. One file holds many games, each stored
// as the minimum needed to faithfully replay every state: the initial racks and
// the full move sequence, each move bundled with the tiles drawn right after
// it. The DataLoader replays a game to materialize its positions on the fly,
// which is roughly 20x smaller on disk than a fully-expanded per-position
// record -- so ~20x more games fit its memory budget, for richer shuffling.
//
// File layout
// -----------
//   [FileHeader              16 B]
//   [GameMetadata  num_games 24 B]
//   For each game g in [0, num_games):
//     [InitialRacks                       16 B]
//     [TurnBlob       num_turns(g)        24 B each]
//
// Each GameMetadata's start_offset points at that game's InitialRacks, with the
// TurnBlob array at (start_offset + sizeof(InitialRacks)).
//
// A game contributes one training row per *eligible* turn, the contiguous
// region the writer deemed trainable; FileHeader's num_sample_positions is
// their sum across the file, so the loader knows the epoch size without
// scanning the body. The single sampled_turn is a separate, eval-only
// convenience for probes and position dumps.

#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

struct GameLog;         // forward (defined in game.h)
struct GameLogStorage;  // forward (defined in game.h)
struct TurnRecord;      // forward (defined in game.h)

namespace binlog {

// "SLOG" in little-endian (bytes 'S','L','O','G' on disk).
inline constexpr uint32_t kMagic = 0x474F4C53u;
inline constexpr uint16_t kVersion = 10;

// FileHeader::flags bits, recording the rules the games were played under so a
// consumer can refuse a corpus that does not suit it. The field occupies what
// was padding, and no flag set -- which is what every file written before the
// bits existed carries -- says standard Scrabble, which those games truthfully
// were. So the version does not move and old corpora stay readable.
inline constexpr uint16_t kFlagFaceUpLeaves = 1u;  // players saw each other'"'"'s leaves

#pragma pack(push, 1)

struct FileHeader {
  uint32_t magic;    // kMagic
  uint16_t version;  // kVersion
  uint16_t flags;    // kFlag* bits; see kFlagFaceUpLeaves
  uint32_t num_games;
  uint32_t num_sample_positions;  // total training rows
};
static_assert(sizeof(FileHeader) == 16, "FileHeader must be 16 bytes");

struct GameMetadata {
  uint64_t start_offset;  // file offset of this game's InitialRacks blob
  uint32_t num_turns;     // length of the TurnBlob array
  uint16_t sampled_turn;  // eval-only; training uses the eligible region below
  int16_t final_score_p0;
  int16_t final_score_p1;
  int16_t initial_score_p0;  // starting score, 0 unless handicapped; the replay
  int16_t initial_score_p1;  // decoder seeds its accumulator from these
  // Training-eligible turn region [eligible_begin, eligible_end), fixed at
  // write time and always non-empty on disk (the writer drops games whose
  // region is empty):
  //   - eligible_end is the pre-endgame prefix length -- the turns whose bag
  //     had tiles when they began, which the non-increasing bag makes a leading
  //     prefix. The field width caps it at 255, excluding turns past that,
  //     reachable only in degenerate pass-heavy games near the 400-turn cap.
  //   - eligible_begin is the position after the game's last random-opening
  //     ply (see Game::set_random_opening). Earlier positions have a
  //     uniformly-random move played after them, which would pollute their
  //     final-score targets; only agent play follows this one.
  uint8_t eligible_begin;
  uint8_t eligible_end;
};
static_assert(sizeof(GameMetadata) == 24, "GameMetadata must be 24 bytes");

// The tiles dealt to each player before play starts.
struct InitialRacks {
  Rack p0;  // 8 B
  Rack p1;  // 8 B
};
static_assert(sizeof(InitialRacks) == 16, "InitialRacks must be 16 bytes");

// One on-disk turn: the move played, plus the tiles drawn immediately after it
// resolved.
struct TurnBlob {
  Move move;   // 16 B (alignof=2; sizeof statically asserted)
  Rack drawn;  // 8 B
};
static_assert(sizeof(TurnBlob) == 24, "TurnBlob must be 24 bytes");

#pragma pack(pop)

// A non-owning view over game `game_idx` of a loaded .slog buffer, filling
// `scratch` with that game's turns and valid until `scratch` is next reused.
// The caller must have validated the file's magic and version.
GameLog make_game_view(const char* buf, uint32_t game_idx, std::vector<TurnRecord>& scratch,
                       uint32_t* sampled_turn);

// A game's training-eligible turn region. May be empty (begin >= end) for a
// degenerate game, which must then be excluded from any training output.
struct EligibleSpan {
  int begin;
  int end;
};

// See GameMetadata's eligible_begin/eligible_end for what the bounds mean.
EligibleSpan eligible_span(const GameLog& log);

// A turn index drawn uniformly within eligible_span(log), or -1 iff the span is
// empty. Taking a GameLog view lets the disk writer and the streaming producer
// share it.
int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng);

// Uniform over ALL turns, endgame positions included; -1 iff the game has none.
// For tasks that sample the whole game.
int pick_any_turn(const GameLog& log, std::mt19937_64& rng);

// Accumulates games from one or more producer threads and flushes them to .slog
// files in fixed-size batches. A flush drops the lock before serializing and
// writing, keeping the I/O off the critical path.
class BinaryLogWriter {
 public:
  // `flags` are the kFlag* bits recorded in every file this writer produces.
  BinaryLogWriter(const std::string& dir, int games_per_file, uint16_t flags = 0);
  ~BinaryLogWriter();  // flushes any pending games

  BinaryLogWriter(const BinaryLogWriter&) = delete;
  BinaryLogWriter& operator=(const BinaryLogWriter&) = delete;

  // Takes ownership by move. May trigger a flush. Thread-safe.
  void append(GameLogStorage&& log);

  // Thread-safe.
  void flush();

 private:
  void write_batch(std::vector<GameLogStorage>&& games);

  std::string dir_;
  int games_per_file_;
  uint16_t flags_;
  std::mutex mutex_;
  std::vector<GameLogStorage> pending_;
};

}  // namespace binlog
}  // namespace scribblez
