#pragma once

// The hasty-rollout task as both the CPU reference and the CUDA kernel see
// it: plain structs with no engine dependencies, so one header compiles under
// g++ (C++26) and nvcc alike. A rollout from a position: draw the opponent's
// rack and our refill from the unseen pool, the opponent plays its top-equity
// move, we play ours, and the outcome is the score differential with the
// play-out bonus. Both sides draw with the same counter-based generator, so
// rollout i holds the same racks on either, and the outcomes compare exactly.
// docs/plans/gpu_hasty_rollouts.md is the design.

#include <cstdint>

#if defined(__CUDACC__)
#define SCRIBBLEZ_HD __host__ __device__
#else
#define SCRIBBLEZ_HD
#endif

namespace scribblez::rollout_abi {

constexpr int kBoardSize = 15;
constexpr int kSquares = kBoardSize * kBoardSize;
constexpr int kRackSize = 7;
constexpr int kTileKinds = 27;  // A..Z, then the blank (26)
constexpr int kBlankTile = 26;
// The unseen pool (at most 100 - 1 tiles) plus the tiles two exchanges can
// return to it.
constexpr int kBagCap = 100 + 2 * kRackSize;

// Glyph codes as Board stores them: 0 empty, 1..26 a natural letter,
// 27..52 a blank designated A..Z.
SCRIBBLEZ_HD inline int glyph_letter(uint8_t code) { return code <= 26 ? code - 1 : code - 27; }
SCRIBBLEZ_HD inline bool glyph_is_blank(uint8_t code) { return code >= 27; }

// A move in Move's own field layout (type, direction, lane, placed glyphs,
// placed-square mask, score), so a device move and a host Move compare
// field by field. An exchange lists its glyphs ascending by tile (A..Z, then
// the blank as code 53), as Move::exchange does.
struct PackedMove {
  uint8_t type = 2;  // 0 PLAY, 1 EXCHANGE, 2 PASS
  uint8_t horizontal = 1;
  uint8_t start = 0;
  uint8_t num_glyphs = 0;
  uint8_t glyphs[kRackSize] = {};
  uint8_t pad = 0;
  uint16_t square_mask = 0;
  uint16_t score = 0;
};

// One rollout's result. Player 0 is us (the leave holder), player 1 the
// opponent, who moves first. `moves[0]` is the opponent's, `moves[1]` ours;
// when the opponent plays out, ours is a PASS and `game_over` is set after
// the first ply.
struct RolloutResult {
  int16_t diff = 0;  // our score minus the opponent's, after the rollout
  uint8_t game_over = 0;
  uint8_t pad = 0;
  PackedMove moves[2];
};

// A rollout's end state, left in device memory for the leaf encoder kernel,
// which builds the leaf eval model's input rows from it there; nothing comes
// back to the host. Player 0 is us, 1 the opponent.
struct LeafState {
  uint8_t squares[kSquares];
  uint8_t racks[2][kTileKinds];   // after the refills
  uint8_t leaves[2][kTileKinds];  // after each player's last move, before its draw
  int scores[2];
  uint8_t bag_size;
  uint8_t game_over;
  uint8_t pad[2];
  uint32_t cross_mask[2][kSquares];  // per view, as the rollout left them
};

// The position, flattened. Cross-check arrays are per view (0 = the natural
// frame's rows, 1 = its columns), in view coordinates, as Board keeps them.
struct PositionAbi {
  uint8_t squares[kSquares] = {};
  uint32_t cross_mask[2][kSquares] = {};
  int16_t cross_score[2][kSquares] = {};
  uint8_t cross_has_neighbor[2][kSquares] = {};
  uint8_t leave[kTileKinds] = {};  // our leave, as counts
  uint8_t pool[kBagCap] = {};      // the unseen tiles, ascending by tile
  int pool_size = 0;
  int num_tiles = 0;  // tiles on the board
  int scores[2] = {0, 0};
  uint64_t seed = 0;
};

// ---- The draw ----
// Philox4x32-10 (Salmon et al.), keyed by the batch seed and counted by
// (rollout, block): random number k of rollout i is output k % 4 of block
// k / 4.

SCRIBBLEZ_HD inline uint32_t philox_mulhi(uint32_t a, uint32_t b) {
  return uint32_t((uint64_t(a) * uint64_t(b)) >> 32);
}

SCRIBBLEZ_HD inline void philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                       uint32_t k0, uint32_t k1, uint32_t out[4]) {
  for (int r = 0; r < 10; ++r) {
    const uint32_t hi0 = philox_mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = philox_mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  out[0] = c0;
  out[1] = c1;
  out[2] = c2;
  out[3] = c3;
}

SCRIBBLEZ_HD inline uint32_t rollout_rand(uint64_t seed, uint32_t rollout, uint32_t k) {
  uint32_t out[4];
  philox4x32_10(rollout, k >> 2, 0u, 0u, uint32_t(seed), uint32_t(seed >> 32), out);
  return out[k & 3u];
}

// The bag as a rollout sees it: the pool's tiles, drawn from the front by a
// partial Fisher-Yates shuffle, with exchanged tiles appended at the back.
struct Bag {
  uint8_t tiles[kBagCap];
  int len = 0;
  int cursor = 0;  // tiles[0, cursor) have been drawn
  int next_rand = 0;

  SCRIBBLEZ_HD int remaining() const { return len - cursor; }

  SCRIBBLEZ_HD uint8_t draw(uint64_t seed, uint32_t rollout) {
    const int k = cursor;
    const int r = k + int(rollout_rand(seed, rollout, uint32_t(next_rand++)) % uint32_t(len - k));
    const uint8_t t = tiles[r];
    tiles[r] = tiles[k];
    tiles[k] = t;
    ++cursor;
    return t;
  }

  SCRIBBLEZ_HD void put_back(uint8_t t) { tiles[len++] = t; }
};

// The rack sizes the draw produces: the opponent's rack, then our refill.
SCRIBBLEZ_HD inline int opp_rack_draw(int pool_size) {
  return pool_size < kRackSize ? pool_size : kRackSize;
}
SCRIBBLEZ_HD inline int our_refill_draw(int pool_size, int leave_size) {
  const int bag = pool_size - opp_rack_draw(pool_size);
  const int want = kRackSize - leave_size;
  return bag < want ? bag : want;
}

}  // namespace scribblez::rollout_abi
