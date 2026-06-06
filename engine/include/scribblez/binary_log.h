#pragma once

// Binary game-log format for training. One file holds many games (configurable
// via --games-per-file); within a file, each game contributes K self-contained
// PositionRecords (K = --samples-per-game). The format is designed for a
// custom C++ data loader that reinterpret_casts straight into memory-mapped
// or read() buffers (everything is fixed-size POD, naturally aligned).
//
// File layout
// -----------
//   [FileHeader              16 B]
//   [GameMetadata  num_games 32 B]
//   [PositionRecord blobs, grouped by game, back-to-back]
//
// Each GameMetadata's start_offset / data_size point at that game's contiguous
// run of (num_positions * sizeof(PositionRecord)) bytes inside the blob region.
//
// Why self-contained position records (no shared "history" buffer)?
// AlphaZeroArcade has to materialize a sliding window of past frames at
// training time (to feed a stack-of-frames input encoder), so its on-disk
// records carry only the minimal per-frame delta and the loader walks back via
// per-frame offsets. Our win-probability model takes a single board snapshot
// plus the last opponent move, so embedding the full state in every sampled
// record costs ~300 B and removes all per-frame indirection.

#include "scribblez/glyph.h"
#include "scribblez/move.h"
#include "scribblez/rack.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace scribblez {

struct GameLog;  // forward (defined in game.h)

namespace binlog {

// "SLOG" in little-endian (bytes 'S','L','O','G' on disk).
inline constexpr uint32_t kMagic = 0x474F4C53u;
inline constexpr uint16_t kVersion = 1;

// Position kind. PLAY turns produce BOTH a pre-move and a post-move record;
// EXCHANGE / PASS turns produce only a pre-move record.
enum class PositionKind : uint8_t {
  kPreMove = 0,   // active player is about to play; full rack
  kPostMove = 1,  // active player just played; rack has played-tiles removed,
                  // refill has not happened yet (bag composition unchanged
                  // from pre-move)
};

#pragma pack(push, 1)

struct FileHeader {
  uint32_t magic;    // kMagic
  uint16_t version;  // kVersion
  uint16_t reserved;
  uint32_t num_games;      // games in this file
  uint32_t num_positions;  // total sampled positions across all games
};
static_assert(sizeof(FileHeader) == 16, "FileHeader must be 16 bytes");

struct GameMetadata {
  uint64_t start_offset;   // file offset of this game's first PositionRecord
  uint32_t data_size;      // bytes = num_positions * sizeof(PositionRecord)
  uint32_t num_positions;  // sampled positions for this game
  uint64_t seed;           // game seed (for traceability)
  int32_t final_score_p0;
  int32_t final_score_p1;
};
static_assert(sizeof(GameMetadata) == 32, "GameMetadata must be 32 bytes");

struct PositionRecord {
  // --- 16-byte small-field block --------------------------------------------
  uint8_t active_player;  // 0 or 1; POV for label computation
  uint8_t position_kind;  // PositionKind
  int16_t move_number;    // 0-indexed turn index in the game
  int32_t score_active;   // cumulative score for active player at this moment
  int32_t score_opp;      // cumulative score for opponent at this moment
  uint32_t reserved0;

  // --- the last move the OPPONENT played (PASS-typed if game-start) ---------
  Move last_opp_move;  // 16 B (sizeof(Move) statically asserted)

  // --- active player's rack at this moment (partial for post-move PLAY) -----
  Rack active_rack;  // 8 B (sizeof(Rack) statically asserted)

  // --- bag composition: TILE_COUNTS - tiles_on_board - rack[0] - rack[1] ----
  // Index 0..25 = A..Z, 26 = blank. (uint8 suffices: max 12 of any letter.)
  uint8_t bag_counts[27];
  uint8_t reserved1[5];  // pad so board[] starts at offset 72

  // --- the 15x15 board (Glyph = 1 byte) -------------------------------------
  Glyph board[225];

  uint8_t reserved2[7];  // pad to multiple of 16
};
static_assert(sizeof(PositionRecord) == 304, "PositionRecord must be 304 bytes");
static_assert(sizeof(PositionRecord) % 16 == 0, "PositionRecord must be 16-aligned");

#pragma pack(pop)

// Replay a game and extract self-contained PositionRecords for K sampled
// positions. The per-game RNG is seeded deterministically from the game's
// seed so a given log always yields the same samples. Returns at most K
// records (fewer if the game produced fewer than K eligible positions, which
// only happens for pathologically short games).
//
// Eligible positions:
//   - pre-move snapshot before every turn
//   - post-move snapshot after every PLAY turn (skipped for EXCHANGE/PASS)
std::vector<PositionRecord> extract_positions(const GameLog& log, int samples_per_game);

// Thread-safe writer that accumulates GameLog objects from one or more
// GameRunner threads and flushes them to .slog files as fixed-size batches.
//
// On flush, the writer drops the lock, then extracts records and writes the
// file off the critical path so other threads aren't blocked by I/O.
class BinaryLogWriter {
 public:
  BinaryLogWriter(const std::string& dir, int games_per_file, int samples_per_game);
  ~BinaryLogWriter();  // flushes any pending games

  BinaryLogWriter(const BinaryLogWriter&) = delete;
  BinaryLogWriter& operator=(const BinaryLogWriter&) = delete;

  // Take ownership of one finished game; may trigger a file flush.
  // (Accepts a const reference rather than a sink-by-value because every
  // current caller has a non-moveable lvalue; the copy is made internally.)
  // Thread-safe.
  void append(const GameLog& log);

  // Force a write of any pending games (no-op if pending is empty).
  // Thread-safe.
  void flush();

 private:
  void write_batch(std::vector<GameLog>&& games);

  std::string dir_;
  int games_per_file_;
  int samples_per_game_;
  std::mutex mutex_;
  std::vector<GameLog> pending_;
};

}  // namespace binlog
}  // namespace scribblez
