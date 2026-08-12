#pragma once

// Hand-built games for tests: an explicit PLAY Move, and a single game
// serialized into an in-memory .slog buffer the real BlockDecoder reads --
// together, what a test needs to put the training replay path and an agent's
// own encoder on the same position.

#include "data/binary_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace scribblez::testing {

// A PLAY Move with an explicit per-tile layout. `rel_mask` is relative to the
// first lane cell (bit 0 == the start cell); `gs` are the placed glyphs in word
// order (count == popcount(rel_mask)).
inline Move make_play_full(int row, int col, bool horizontal, uint16_t rel_mask, uint16_t score,
                           std::initializer_list<Glyph> gs) {
  std::array<Glyph, RACK_SIZE> played{};
  int n = 0;
  for (Glyph g : gs) {
    if (n >= RACK_SIZE) break;
    played[n++] = g;
  }
  const int lane0 = horizontal ? col : row;
  const int start = horizontal ? row : col;
  const uint16_t mask = static_cast<uint16_t>(rel_mask << lane0);
  return Move::play(horizontal, start, mask, score, played.data(), n);
}

// Append the raw bytes of a trivially-copyable value to a byte buffer.
template <class T>
void append_pod(std::vector<char>& buf, const T& v) {
  const char* p = reinterpret_cast<const char*>(&v);
  buf.insert(buf.end(), p, p + sizeof(T));
}

// FileHeader, one GameMetadata (with a caller-chosen sampled_turn), then the
// game's InitialRacks and TurnBlob[]. FINAL scores are left at zero: a fixture
// built this way is for comparing input rows, not the score-derived targets.
// `initial_scores` is the head-start handicap the replay decoder seeds its
// score accumulator from -- {0, 0} for an ordinary game.
inline std::vector<char> build_slog(const binlog::InitialRacks& ir,
                                    const std::vector<binlog::TurnBlob>& turns,
                                    uint32_t sampled_turn,
                                    std::array<int, 2> initial_scores = {0, 0}) {
  binlog::FileHeader hdr{};
  hdr.magic = binlog::kMagic;
  hdr.version = binlog::kVersion;
  hdr.num_games = 1;

  binlog::GameMetadata gm{};
  gm.start_offset = sizeof(binlog::FileHeader) + sizeof(binlog::GameMetadata);
  gm.num_turns = static_cast<uint32_t>(turns.size());
  gm.sampled_turn = sampled_turn;
  gm.initial_score_p0 = static_cast<int16_t>(initial_scores[0]);
  gm.initial_score_p1 = static_cast<int16_t>(initial_scores[1]);

  std::vector<char> buf;
  append_pod(buf, hdr);
  append_pod(buf, gm);
  append_pod(buf, ir);
  for (const binlog::TurnBlob& t : turns) append_pod(buf, t);
  return buf;
}

}  // namespace scribblez::testing
