#pragma once

// The .slog binary game-log format used for training data. A file holds many
// games, each stored as the minimum needed to replay every state: the initial
// racks and the move sequence, each move bundled with the tiles drawn right
// after it. The DataLoader replays a game to materialize its positions on the
// fly. That is roughly 20x smaller on disk than storing each position expanded,
// so ~20x more games fit the loader's memory budget and shuffling is richer.
// docs/architecture.md covers where the format sits in the pipeline.
//
// File layout
// -----------
//   [FileHeader                            16 B]
//   [GameMetadata  x num_games             24 B each]
//   For each game g in [0, num_games):
//     [InitialRacks                        16 B]  <- GameMetadata::start_offset
//     [TurnBlob      x num_turns(g)        24 B each]
//
// A game contributes one training row per turn of its eligible region (see
// GameMetadata). FileHeader::num_sample_positions is the sum of those region
// lengths, so a reader knows the epoch size without scanning the body.

#include "game/move.h"
#include "game/rack.h"

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace scribblez {

struct GameLog;  // game/game_log.h
struct GameLogStorage;
struct TurnRecord;

namespace binlog {

// "SLOG" in little-endian (bytes 'S','L','O','G' on disk).
inline constexpr uint32_t kMagic = 0x474F4C53u;
inline constexpr uint16_t kVersion = 10;

// FileHeader::flags bits record the rules the games were played under, so a
// consumer can refuse a corpus that does not suit it.
inline constexpr uint16_t kFlagFaceUpLeaves = 1u;  // players saw each other's leaves

#pragma pack(push, 1)

struct FileHeader {
  uint32_t magic;    // kMagic
  uint16_t version;  // kVersion
  uint16_t flags;
  uint32_t num_games;
  uint32_t num_sample_positions;  // total training rows
};
static_assert(sizeof(FileHeader) == 16, "FileHeader must be 16 bytes");

struct GameMetadata {
  uint64_t start_offset;  // file offset of this game's InitialRacks blob
  uint32_t num_turns;     // length of the TurnBlob array
  // One turn drawn uniformly from the eligible region, for eval-only callers
  // (probes, position dumps) that want a single position per game.
  uint16_t sampled_turn;
  int16_t final_score_p0;
  int16_t final_score_p1;
  // Starting scores: 0 unless the game was handicapped. Replay seeds its
  // running scores from these.
  int16_t initial_score_p0;
  int16_t initial_score_p1;
  // The training-eligible turn region [eligible_begin, eligible_end). The
  // writer drops games whose region is empty, so it is never empty on disk.
  //   - eligible_end: the number of leading turns that began with tiles in the
  //     bag. The bag never grows, so these form a prefix. The uint8_t width
  //     caps it at 255; only degenerate pass-heavy games near the 400-turn cap
  //     get that long.
  //   - eligible_begin: the turn of the last random-opening ply (see
  //     Game::set_random_opening), or 0 without one. Training samples
  //     post-move positions by default, and the position after this ply is
  //     the first that only agent play follows. An earlier one has a random
  //     move ahead of it, which would pollute its final-score target.
  uint8_t eligible_begin;
  uint8_t eligible_end;
};
static_assert(sizeof(GameMetadata) == 24, "GameMetadata must be 24 bytes");

struct InitialRacks {
  Rack p0;  // 8 B
  Rack p1;  // 8 B
};
static_assert(sizeof(InitialRacks) == 16, "InitialRacks must be 16 bytes");

// One turn: the move played and the tiles its player drew right after it.
struct TurnBlob {
  Move move;   // 16 B
  Rack drawn;  // 8 B
};
static_assert(sizeof(TurnBlob) == 24, "TurnBlob must be 24 bytes");

#pragma pack(pop)

// A view of game `game_idx` in a loaded .slog buffer. Its records live in
// `scratch`, so the view is valid until `scratch` is next reused. Only each
// record's move and draw are filled in (see complete_turn_records). The caller
// must have validated the file's magic and version. `sampled_turn`, if
// non-null, receives the game's GameMetadata::sampled_turn.
GameLog make_game_view(const char* buf, uint32_t game_idx, std::vector<TurnRecord>& scratch,
                       uint32_t* sampled_turn);

// Fills in the rest of the first `num_turns` records of a make_game_view view:
// each turn's player, pre-move rack, bag size, score delta and running scores,
// derived by replay. Training does not need this, since its encoder re-derives
// the state itself; a GCG export needs whole records.
void complete_turn_records(const GameLog& g, int num_turns, std::vector<TurnRecord>& scratch);

// A game's training-eligible turn region, with the bounds described at
// GameMetadata::eligible_begin/eligible_end. May be empty (begin >= end) for a
// degenerate game, which must then be excluded from any training output.
struct EligibleSpan {
  int begin;
  int end;
};

EligibleSpan eligible_span(const GameLog& log);

// A turn drawn uniformly from eligible_span(log), or -1 iff the span is empty.
int pick_sampled_turn(const GameLog& log, std::mt19937_64& rng);

// A turn drawn uniformly from all turns, endgame included, or -1 iff the game
// has none. For tasks that sample the whole game.
int pick_any_turn(const GameLog& log, std::mt19937_64& rng);

// Accumulates games from any number of producer threads and writes them out as
// .slog files of `games_per_file` games each. Serialization and I/O happen
// outside the lock, so producers are not blocked on disk.
class BinaryLogWriter {
 public:
  BinaryLogWriter(const std::string& dir, int games_per_file, uint16_t flags = 0);
  ~BinaryLogWriter();  // flushes any pending games

  BinaryLogWriter(const BinaryLogWriter&) = delete;
  BinaryLogWriter& operator=(const BinaryLogWriter&) = delete;

  // Thread-safe. Writes a file once `games_per_file` games are pending.
  void append(GameLogStorage&& log);

  // Thread-safe. Writes all pending games, even if fewer than a full file.
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
