#pragma once

// Binary game-log format for training. One file holds many games (configurable
// via --games-per-file). Each game is stored as the minimum data needed to
// faithfully replay every state: the initial racks dealt to each player and
// the full sequence of moves they played (each move bundled with the tiles
// drawn from the bag right after that move).
//
// The DataLoader at training time replays each game with a GameStateEncoder
// to materialize every eligible position on-the-fly. Compared to storing a
// fully-expanded per-position record, this is roughly 20x smaller on disk,
// which translates to ~20x more games resident in the DataLoader's memory
// budget for richer shuffling.
//
// File layout
// -----------
//   [FileHeader              16 B]
//   [GameMetadata  num_games 24 B]
//   For each game g in [0, num_games):
//     [InitialRacks                       16 B]
//     [TurnBlob       num_turns(g)        24 B each]
//
// Each GameMetadata's start_offset points at that game's InitialRacks; the
// TurnBlob array starts at (start_offset + sizeof(InitialRacks)).
//
// GameMetadata carries both the final scores and the per-player starting
// scores. The starting scores are normally {0, 0} but can be a head-start
// handicap; the replay decoder seeds its score accumulator from them so the
// score-differential input feature reflects the handicap at every position.

#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

struct GameLog;         // forward (defined in game.h)
struct GameLogStorage;  // forward (defined in game.h)

namespace binlog {

// "SLOG" in little-endian (bytes 'S','L','O','G' on disk).
inline constexpr uint32_t kMagic = 0x474F4C53u;
inline constexpr uint16_t kVersion = 8;

#pragma pack(push, 1)

struct FileHeader {
  uint32_t magic;    // kMagic
  uint16_t version;  // kVersion
  uint16_t reserved;
  uint32_t num_games;  // games in this file == sample positions in this
                       // file (one sample per game, chosen at write-time)
  uint32_t reserved2;
};
static_assert(sizeof(FileHeader) == 16, "FileHeader must be 16 bytes");

struct GameMetadata {
  uint64_t start_offset;  // file offset of this game's InitialRacks blob
  uint32_t num_turns;     // length of the TurnBlob array
  uint32_t sampled_turn;  // 0-based turn index chosen for this game's sample
                          // (pre-move snapshot encodes state AT this turn;
                          // post-move snapshot encodes state AFTER this
                          // turn's move is applied, before draw). Chosen
                          // uniformly at write-time among turns where the
                          // bag has > 0 tiles when the turn begins.
  int16_t final_score_p0;
  int16_t final_score_p1;
  int16_t initial_score_p0;  // per-player starting score (0 unless handicapped)
  int16_t initial_score_p1;
};
static_assert(sizeof(GameMetadata) == 24, "GameMetadata must be 24 bytes");

// Per-game initial state: the tiles dealt to each player before play starts.
// Trailing entries of each rack slot are empty Tiles when the bag was starved
// and a player received fewer than RACK_SIZE tiles.
struct InitialRacks {
  Rack p0;  // 7 B
  Rack p1;  // 7 B
};
static_assert(sizeof(InitialRacks) == 16, "InitialRacks must be 16 bytes");

// One on-disk turn: the move that was played, plus the tiles drawn from the
// bag immediately after the move resolved (and rack mutations applied).
// Trailing entries of `drawn` are empty Tiles.
struct TurnBlob {
  Move move;   // 16 B (alignof=2; sizeof statically asserted)
  Rack drawn;  // 8 B
};
static_assert(sizeof(TurnBlob) == 24, "TurnBlob must be 24 bytes");

#pragma pack(pop)

// Pick a turn index for `log` uniformly at random among turns whose
// bag_size_before > 0 (i.e. pre-endgame positions, where one position is
// sampled per game). Returns -1 iff no eligible turn exists (a degenerate
// game), in which case the game must be excluded from any output. Operates on a
// GameLog view, so it serves both the disk writer and the streaming producer.
int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng);

// Thread-safe writer that accumulates GameLog objects from one or more
// GameRunner threads and flushes them to .slog files as fixed-size batches.
//
// On flush, the writer drops the lock and then serializes + writes the file
// off the critical path so other threads aren't blocked by I/O.
class BinaryLogWriter {
 public:
  BinaryLogWriter(const std::string& dir, int games_per_file);
  ~BinaryLogWriter();  // flushes any pending games

  BinaryLogWriter(const BinaryLogWriter&) = delete;
  BinaryLogWriter& operator=(const BinaryLogWriter&) = delete;

  // Take ownership of one finished game by moving its log storage in (no copy);
  // may trigger a file flush. Thread-safe.
  void append(GameLogStorage&& log);

  // Force a write of any pending games (no-op if pending is empty).
  // Thread-safe.
  void flush();

 private:
  void write_batch(std::vector<GameLogStorage>&& games);

  std::string dir_;
  int games_per_file_;
  std::mutex mutex_;
  std::vector<GameLogStorage> pending_;
};

}  // namespace binlog
}  // namespace scribblez
