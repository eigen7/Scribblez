#include "scribblez/game_state_encoder.h"

#include "scribblez/glyph.h"
#include "scribblez/input_encoder.h"
#include "scribblez/tile.h"

#include <cassert>
#include <cstring>

namespace scribblez {
namespace binlog {

namespace {

// Index into a single 15x15 plane: row-major if !flip, transposed if flip.
inline int plane_idx(int r, int c, bool flip) {
  return flip ? (c * kBoardSide + r) : (r * kBoardSide + c);
}

// Plane offsets within the spatial block.
inline constexpr int kBlankPlane = kLetterPlanes;                          // 26
inline constexpr int kPremiumPlane0 = kLetterPlanes + kBlankMarkerPlanes;  // 27
inline constexpr int kLastOppPlane = kPremiumPlane0 + kPremiumPlanes;      // 31

// Letter, blank-marker, and premium planes.
void encode_board_planes(const Board& board, bool flip, float* planes_out) {
  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      const Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      const int letter = g.letter().index();  // 0..25, valid for any non-empty glyph
      planes_out[letter * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
      if (g.is_blank()) {
        planes_out[kBlankPlane * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
      }
    }
  }

  const auto& prem = Board::PREMIUM;
  for (int r = 0; r < kBoardSide; ++r) {
    for (int c = 0; c < kBoardSide; ++c) {
      const Premium p = prem[r * kBoardSide + c];
      if (p == Premium::NONE) continue;
      int offset = -1;
      if (p == Premium::DLS)
        offset = 0;
      else if (p == Premium::TLS)
        offset = 1;
      else if (p == Premium::DWS)
        offset = 2;
      else if (p == Premium::TWS)
        offset = 3;
      if (offset < 0) continue;
      planes_out[(kPremiumPlane0 + offset) * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
    }
  }
}

// Last-opp-placement plane: mark squares the opponent placed tiles on
// during their most recent turn. Uses Move::square_mask -- bit i is set
// iff the i-th cell along the move direction was a newly placed tile.
void encode_last_opp_plane(const Move& m, bool flip, float* planes_out) {
  if (m.type != MoveType::PLAY) return;  // EXCHANGE / PASS -> all zeros
  const int dr = m.horizontal ? 0 : 1;
  const int dc = m.horizontal ? 1 : 0;
  int r = m.start_row, c = m.start_col;
  uint16_t mask = m.square_mask;
  for (int i = 0; i < kBoardSide; ++i) {
    if (r < 0 || r >= kBoardSide || c < 0 || c >= kBoardSide) break;
    if (mask == 0) break;
    if (mask & 1u) {
      planes_out[kLastOppPlane * kBoardCells + plane_idx(r, c, flip)] = 1.0f;
    }
    mask = static_cast<uint16_t>(mask >> 1);
    r += dr;
    c += dc;
  }
}

// Fill `out` with the active player's "unseen pool" composition:
//   unseen[i] = TILE_COUNTS[i] - (#tile i on board) - (#tile i in my_rack)
// Every tile is in exactly one of (bag, board, p0 rack, p1 rack); the
// active player has no way to distinguish the bag from the opponent's
// rack, so the two are lumped together as the "unseen pool". This
// depends only on data the active player observes.
void compute_unseen_pool(uint8_t out[27], const Board& board, const Rack& my_rack) {
  for (int i = 0; i < 27; ++i) out[i] = static_cast<uint8_t>(TILE_COUNTS[i]);
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      Glyph g = board.at(r, c);
      if (g.is_empty()) continue;
      Tile t = g.rack_tile();
      assert(out[t] > 0);
      --out[t];
    }
  }
  for (Tile t : my_rack.tiles()) {
    if (t.is_empty()) continue;
    assert(out[t] > 0);
    --out[t];
  }
}

// All 58 scalar features. Raw counts/values throughout; the model handles
// normalization.
void encode_scalars(const Rack& my_rack, const uint8_t unseen_pool[27], const Move& last_opp_move,
                    int score_active, int score_opp, float* out) {
  // Active-rack histogram (raw counts).
  int rack_hist[27] = {0};
  for (Tile t : my_rack.tiles()) {
    if (!t.is_empty()) ++rack_hist[t.index()];
  }
  float* p = out;
  for (int i = 0; i < 27; ++i) *p++ = static_cast<float>(rack_hist[i]);

  // Unseen-pool composition (raw counts).
  int unseen_total = 0;
  for (int i = 0; i < 27; ++i) {
    *p++ = static_cast<float>(unseen_pool[i]);
    unseen_total += unseen_pool[i];
  }

  // Misc scalars (raw). Note: opp_rack_size is intentionally absent --
  // it equals min(unseen_total, 7) under Scrabble's refill rule, so the
  // model can derive it from the unseen-pool scalars.
  *p++ = static_cast<float>(score_active - score_opp);
  *p++ = static_cast<float>(unseen_total);
  *p++ = static_cast<float>(my_rack.size());

  // Last opponent move: num_glyphs only. (Move type is recoverable as
  // num_glyphs==0 -> PASS, num_glyphs>0 with empty placement plane ->
  // EXCHANGE, num_glyphs>0 with nonzero placement plane -> PLAY.)
  *p++ = static_cast<float>(last_opp_move.num_glyphs());
}

// Shared back-end for both pre-move and post-PLAY encoding. Takes only
// POV-visible inputs.
void encode_pov(const Board& board, const Rack& my_rack, const Move& last_opp_move,
                int score_active, int score_opp, bool apply_flip, float* out) {
  std::memset(out, 0, sizeof(float) * static_cast<size_t>(kInputFloats));
  encode_board_planes(board, apply_flip, out);
  encode_last_opp_plane(last_opp_move, apply_flip, out);
  uint8_t unseen[27];
  compute_unseen_pool(unseen, board, my_rack);
  encode_scalars(my_rack, unseen, last_opp_move, score_active, score_opp, out + kSpatialFloats);
}

void remove_glyph_tiles_from_rack(Rack& rack, const Move& m) {
  const int n = m.num_glyphs();
  for (int i = 0; i < n; ++i) {
    [[maybe_unused]] bool ok = rack.remove(m.glyphs[i].rack_tile());
    assert(ok);
  }
}

}  // namespace

void GameStateEncoder::apply_move(const Move& move) {
  if (move.type == MoveType::PLAY) {
    board_.apply(move);
    scores_[active_] += move.score;
  }
  // EXCHANGE / PASS: board and score are unchanged.
  last_move_by_[active_] = move;
  active_ = 1 - active_;
  ++turn_index_;
}

void GameStateEncoder::encode_input(const Rack& my_rack, bool apply_flip, float* out) const {
  const int opp = 1 - active_;
  encode_pov(board_, my_rack, last_move_by_[opp], scores_[active_], scores_[opp], apply_flip, out);
}

void GameStateEncoder::encode_input_post_play(const Move& play_move, const Rack& my_rack,
                                              bool apply_flip, float* out) const {
  assert(play_move.type == MoveType::PLAY);
  const int opp = 1 - active_;

  // Materialize the transient post-PLAY state without mutating *this*.
  Board board_post = board_;
  board_post.apply(play_move);

  Rack rack_post = my_rack;
  remove_glyph_tiles_from_rack(rack_post, play_move);

  const int score_active_post = scores_[active_] + play_move.score;

  // From the active player's POV the most recent opponent move is unchanged
  // (the opp hasn't moved since their previous turn).
  encode_pov(board_post, rack_post, last_move_by_[opp], score_active_post, scores_[opp], apply_flip,
             out);
}

}  // namespace binlog
}  // namespace scribblez
