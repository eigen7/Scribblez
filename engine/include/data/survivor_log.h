#pragma once

// The .survivors sidecar: the second stage of the turns an agent searched in
// self-play (docs/plans/deep_bot_generations.md, "Self-play writes the
// pools"). The agent's screen labels every move of a turn, and that is the
// turn's pool in the same-stem .pool (data/pool_log.h). The moves the screen
// keeps, its survivors, then get the sims the agent decides by, or the
// endgame solver's value where the move empties the bag. This file holds
// what those gave each survivor, so that a trainer can take its target from
// the decision and not from the screen.
//
// Layout (all integers little-endian, the game code's native byte order):
//
//   [SurvivorFileHeader                                     16 B]
//   per position, in the .pool's order:
//     [SurvivorPositionHeader                               16 B]
//     [SurvivorRecord x num_survivors                       20 B each]
//     [float32 x the sum of the survivors' rollouts: each survivor's
//      rollout win values in turn]
//
// A position with no second stage (the agent's reply model keeps none) has
// no survivors.

#include "data/slog_sampling.h"
#include "sim/pool_labels.h"

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace scribblez {

inline constexpr const char* kSurvivorExtension = ".survivors";

// "SURV" in little-endian (bytes 'S','U','R','V' on disk).
inline constexpr uint32_t kSurvivorMagic = 0x56525553u;
// Bump it whenever the layout or a field's meaning changes; readers accept
// only this version.
inline constexpr uint16_t kSurvivorFormatVersion = 1;

// How the opponent replies in the sims of a file's survivors.
enum class SurvivorReply : uint16_t {
  HASTY = 0,  // by HastyBot
};

// What a survivor's value came from.
enum class SurvivorSource : uint8_t {
  // The reply model's sims, raced among the survivors that have them: a pool
  // label over those, and the rollouts' values.
  SIMS = 0,
  // The endgame solver's replies at the racks sampled for the opponent: a
  // win probability alone.
  SOLVED = 1,
};

#pragma pack(push, 1)

struct SurvivorFileHeader {
  uint32_t magic;    // kSurvivorMagic
  uint16_t version;  // kSurvivorFormatVersion
  uint16_t reply;    // SurvivorReply
  uint32_t num_positions;
  uint32_t rollouts;  // the sims' budget per survivor; racing stops some early
};
static_assert(sizeof(SurvivorFileHeader) == 16, "SurvivorFileHeader must be 16 bytes");

struct SurvivorPositionHeader {
  uint32_t game_index;  // game within the companion .slog file
  uint32_t turn_index;
  uint32_t num_survivors;
  uint32_t reserved;  // 0
};
static_assert(sizeof(SurvivorPositionHeader) == 16, "SurvivorPositionHeader must be 16 bytes");

struct SurvivorRecord {
  uint32_t candidate;  // the survivor's index among its position's pool candidates
  float win_prob;      // the sims' mean win value, or the solver's value
  float paired_se;     // PoolLabel::paired_se among the simmed survivors; 0 if solved
  float q;             // PoolLabel::q among the simmed survivors; 0 if solved
  uint16_t rollouts;   // completed, the values the block holds; 0 if solved
  uint8_t source;      // SurvivorSource
  uint8_t reserved;    // 0
};
static_assert(sizeof(SurvivorRecord) == 20, "SurvivorRecord must be 20 bytes");

#pragma pack(pop)

// One survivor, as a writer takes it. `label.rollouts` counts its values;
// a solved survivor's label holds its win probability alone.
struct Survivor {
  uint32_t candidate = 0;
  SurvivorSource source = SurvivorSource::SIMS;
  PoolLabel label;
};

// One position's survivors: `values` holds each one's rollout win values in
// turn, label.rollouts of them.
struct TurnSurvivors {
  binlog::GamePositionIndex at;
  std::vector<Survivor> survivors;
  std::vector<float> values;
};

// Accumulates positions in memory and writes the .survivors file atomically
// on close().
class SurvivorWriter {
 public:
  SurvivorWriter(const std::string& path, SurvivorReply reply, int rollouts);
  // Closes if close() was not called, unless an exception is unwinding: a
  // partial file would pass for a finished one, so it is dropped instead.
  ~SurvivorWriter();

  SurvivorWriter(const SurvivorWriter&) = delete;
  SurvivorWriter& operator=(const SurvivorWriter&) = delete;

  void add_position(const TurnSurvivors& p);
  void close();

 private:
  std::string path_;
  std::vector<char> buffer_;
  uint32_t num_positions_ = 0;
  bool closed_ = false;
  int uncaught_at_open_ = std::uncaught_exceptions();
};

// Loads a whole .survivors file and serves per-position views. Throws
// util::Exception on a missing or truncated file, bad magic, or a format
// version other than kSurvivorFormatVersion.
class SurvivorReader {
 public:
  // Per-position view into the reader's buffer; valid while the reader lives.
  struct Position {
    const SurvivorPositionHeader* header;
    const SurvivorRecord* survivors;  // header->num_survivors
    const float* values;              // each survivor's `rollouts` values in turn
  };

  explicit SurvivorReader(const std::string& path);

  const SurvivorFileHeader& header() const {
    return *reinterpret_cast<const SurvivorFileHeader*>(buffer_.data());
  }
  int num_positions() const { return int(positions_.size()); }
  Position position(int i) const { return positions_[size_t(i)]; }

 private:
  std::vector<char> buffer_;
  std::vector<Position> positions_;
};

}  // namespace scribblez
