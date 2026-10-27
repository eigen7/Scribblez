#pragma once

// The hasty-rollout task as both the CPU reference and the CUDA kernel see
// it: plain structs with no engine dependencies, so one header compiles under
// g++ (C++26) and nvcc alike. A rollout from a start (a candidate applied to
// the board): seat the opponent's known leave if the group has one, shuffle
// the rest of the unseen pool, draw the rest of the opponent's rack and then
// our refill, each player's exchanged tiles going back after its own refill
// (if our move went out with the bag empty, the game is over here), the
// opponent plays its top-equity move, we play ours,
// and the outcome is the score differential with the play-out bonus. If the
// bag is empty after those two plies, both keep playing top-equity moves to
// the game's end, so the outcome is exact rather than a leaf. With the bag
// empty a pass leaves the position unchanged and the players are
// deterministic, so the second scoreless turn in a row ends the game as the
// sixth would (each player loses its rack's value).
// Both sides draw with the same counter-based generator, so rollout i holds
// the same racks on either, and the outcomes compare exactly.
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
// The unseen pool (at most 100 - 1 tiles). An exchange's tiles go back into
// the slots its draw freed, so the bag never outgrows it.
constexpr int kBagCap = 100;
// A bound the rollout never reaches: two plies, then with the bag empty
// every play takes a tile from the racks (2 x kRackSize at most), with at
// most one pass between plays.
constexpr int kMaxRolloutPlies = 2 + 2 * 2 * kRackSize + 1;

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
// opponent, who moves first. `moves[0]` is the opponent's, `moves[1]` ours
// (any later plies of a game played to the end are not reported);
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

// A rollout's start, flattened: a candidate applied to the board, so we
// (player 0, the chooser) have just moved and hold `leave`; the opponent
// moves next. Cross-check arrays are per view (0 = the natural frame's rows,
// 1 = its columns), in view coordinates, as Board keeps them.
struct PositionAbi {
  uint8_t squares[kSquares] = {};
  uint32_t cross_mask[2][kSquares] = {};
  int16_t cross_score[2][kSquares] = {};
  uint8_t cross_has_neighbor[2][kSquares] = {};
  uint8_t leave[kTileKinds] = {};    // our leave, as counts
  uint8_t returned[kRackSize] = {};  // an exchange's tiles, back in the bag after our refill
  uint8_t num_returned = 0;
  // The opponent's exchanged tiles when its last move was an exchange, back in
  // the bag after its refill (its known leave seated, the rest drawn).
  uint8_t opp_returned[kRackSize] = {};
  uint8_t num_opp_returned = 0;
  int num_tiles = 0;  // tiles on the board
  int scores[2] = {0, 0};
  uint32_t group = 0;
};

// What a group of starts shares: the unseen pool as the chooser saw it before
// moving, the seed, and the opponents' known leaves, one per rollout index.
// Rollout i of every start in the group draws the same opponent rack and the
// same bag order, so the starts' outcomes are paired.
struct GroupAbi {
  uint8_t pool[kBagCap] = {};  // ascending by tile
  int pool_size = 0;
  uint64_t seed = 0;
  // Rollout i's opponent leave is leaves[leave_base + i] for i < num_leaves;
  // without one, the opponent's whole rack is drawn.
  uint32_t leave_base = 0;
  uint32_t num_leaves = 0;
};

// An opponent's known leave, up to a full rack (a pass keeps all seven):
// `n` tile codes, ascending.
struct OppLeaveAbi {
  uint8_t tiles[kRackSize] = {};
  uint8_t n = 0;
};

// One rollout of a batch: start `position`, rollout index `rollout` (which
// keys its draws).
struct RolloutTask {
  uint32_t position = 0;
  uint32_t rollout = 0;
};

// ---- The draw ----
// Philox4x32-10 (Salmon et al.), keyed by the batch seed and counted by
// (rollout, block, stream): random number k of a rollout's stream is output
// k % 4 of block k / 4. The deal's shuffle and exchanges' put-backs read
// separate streams, so an exchange shifts no shuffle's numbers.

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

constexpr uint32_t kShuffleStream = 0;
constexpr uint32_t kPutBackStream = 1;

SCRIBBLEZ_HD inline uint32_t rollout_rand(uint64_t seed, uint32_t rollout, uint32_t stream,
                                          uint32_t k) {
  uint32_t out[4];
  philox4x32_10(rollout, k >> 2, stream, 0u, uint32_t(seed), uint32_t(seed >> 32), out);
  return out[k & 3u];
}

// The bag as a rollout sees it: the pool's tiles, shuffled once the deal has
// seated the opponent's known leave, then drawn from both ends, ours (player
// 0) from the front and the opponent's from the back. A player's k-th tile so
// sits at a fixed position however many tiles the other player draws, and
// rollout i of every start in a group shuffles the same pool the same way:
// the starts' draws stay paired after their moves draw different counts.
struct Bag {
  uint8_t tiles[kBagCap];
  int front = 0;  // tiles[front, back) are in the bag
  int back = 0;
  int put_backs = 0;  // the put-back stream's numbers used

  SCRIBBLEZ_HD int remaining() const { return back - front; }

  // Fisher-Yates over the bag.
  SCRIBBLEZ_HD void shuffle(uint64_t seed, uint32_t rollout) {
    for (int i = back - 1; i > front; --i) {
      const uint32_t span = uint32_t(i - front + 1);
      const int j = front + int(rollout_rand(seed, rollout, kShuffleStream, uint32_t(i)) % span);
      const uint8_t t = tiles[i];
      tiles[i] = tiles[j];
      tiles[j] = t;
    }
  }

  SCRIBBLEZ_HD uint8_t draw(int player) { return player == 0 ? tiles[front++] : tiles[--back]; }

  // Returns `t` for `player`, who drew at least as many tiles as it returns
  // since its last put-back: into the slot at its end that a draw freed, then
  // swapped with a uniformly random tile of the bag. Every order stays equally
  // likely, and the other player's tiles keep their positions, one of them
  // replaced by `t`.
  SCRIBBLEZ_HD void put_back(uint8_t t, int player, uint64_t seed, uint32_t rollout) {
    const int end = player == 0 ? --front : back++;
    const uint32_t k = uint32_t(put_backs++);
    const int r =
      front + int(rollout_rand(seed, rollout, kPutBackStream, k) % uint32_t(remaining()));
    tiles[end] = tiles[r];
    tiles[r] = t;
  }
};

// Removes one `tile` from the bag before its shuffle, keeping the rest in
// order: how the opponent's known leave leaves the pool.
SCRIBBLEZ_HD inline void remove_from_bag(Bag& bag, uint8_t tile) {
  for (int i = bag.front; i < bag.back; ++i) {
    if (bag.tiles[i] != tile) continue;
    for (int j = i + 1; j < bag.back; ++j) bag.tiles[j - 1] = bag.tiles[j];
    --bag.back;
    return;
  }
}

// How many tiles a rack of `held` tiles draws from a bag of `remaining`.
SCRIBBLEZ_HD inline int refill_draw(int remaining, int held) {
  const int want = kRackSize - held;
  return remaining < want ? remaining : want;
}

}  // namespace scribblez::rollout_abi
