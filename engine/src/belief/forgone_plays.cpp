#include "belief/forgone_plays.h"

#include "game/movegen.h"
#include "game/rack.h"
#include "game/tile_counts.h"
#include "lexicon/dictionary.h"

#include <algorithm>
#include <climits>

namespace scribblez::belief {

namespace {

bool has_forgone_plays(const Move& move) {
  return move.type() == MoveType::PLAY && move.num_glyphs() < RACK_SIZE;
}

TileCounts rack_tiles_of(const Move& move) {
  TileCounts out;
  for (int i = 0; i < move.num_glyphs(); ++i) out.add(move.glyph(i).rack_tile());
  return out;
}

// Whether the board after `move` holds every copy of each tile kind.
std::array<bool, TILE_KINDS> exhausted_after(const Board& before, const TileCounts& played) {
  const TileCounts on_board = before.tile_counts();
  std::array<bool, TILE_KINDS> out{};
  for (Tile t = Tile::of(0); t < TILE_KINDS; ++t) {
    out[t] = on_board.count(t) + played.count(t) >= TILE_COUNTS[t];
  }
  return out;
}

// The best score offered per tau so far.
class BestByTau {
 public:
  BestByTau() { best_.fill(INT_MIN); }

  void offer(Tile tau, int score) { best_[tau] = std::max(best_[tau], score); }

  ForgonePlays margins(int move_score, const std::array<bool, TILE_KINDS>& exhausted) const;

 private:
  std::array<int, TILE_KINDS> best_;
};

ForgonePlays BestByTau::margins(int move_score,
                                const std::array<bool, TILE_KINDS>& exhausted) const {
  ForgonePlays out = ForgonePlays::none();
  for (int t = 0; t < TILE_KINDS; ++t) {
    if (exhausted[t]) continue;
    const int margin = best_[t] == INT_MIN ? ForgonePlays::kFloor : best_[t] - move_score;
    out.margin[t] = int8_t(std::clamp(margin, ForgonePlays::kFloor, ForgonePlays::kCeiling));
  }
  return out;
}

// The product of the word multipliers under `play`'s placed tiles.
int word_multiplier(const Board& board, const Move& play) {
  int mult = 1;
  visit_placed_squares(play, [&](int r, int c) { mult *= board.premium_at(r, c).word_mult(); });
  return mult;
}

// What a blank of `play`, at lane cell `along` and designated `letter`, would
// add to the play's score as a real `letter` tile: its value on that square, in
// the main word and in any perpendicular word, as build_play scores them.
int designation_gain(const Board& board, const Move& play, int along, Tile letter, int word_mult) {
  const int r = play.horizontal() ? play.start() : along;
  const int c = play.horizontal() ? along : play.start();
  const Premium premium = board.premium_at(r, c);
  const int value = letter.value() * premium.letter_mult();
  const CrossCheck& cross =
    board.cross_checks(!play.horizontal())[play.start() * BOARD_SIZE + along];
  return value * word_mult + (cross.has_neighbor ? value * premium.word_mult() : 0);
}

// Offers `play`, generated from the played tiles `k` plus one blank, to every
// tau it stands for. As a blank play, it needs tau = blank when it places one
// more blank than `k` holds. As a play of letter L, each of its blanks
// designated L may be the extra tile, now a real L, provided the play also
// places every L of `k` as a real tile (with fewer, `k` makes it alone).
void offer_blank_play(const Board& board, const Move& play, const TileCounts& k, BestByTau& best) {
  TileCounts placed;
  for (int i = 0; i < play.num_glyphs(); ++i) placed.add(play.glyph(i).rack_tile());
  if (placed.blanks() > k.blanks()) best.offer(BLANK, play.score());

  const int word_mult = word_multiplier(board, play);
  uint16_t mask = play.square_mask();
  int i = 0;
  for (int along = 0; mask; ++along, mask >>= 1) {
    if ((mask & 1u) == 0) continue;
    const Glyph g = play.glyph(i++);
    if (!g.is_blank() || placed.count(g.letter()) != k.count(g.letter())) continue;
    best.offer(g.letter(),
               play.score() + designation_gain(board, play, along, g.letter(), word_mult));
  }
}

}  // namespace

ForgonePlays ForgonePlays::none() {
  ForgonePlays out;
  out.margin.fill(kNone);
  return out;
}

ForgonePlays compute_forgone_plays(const Board& before, const Dictionary& dict, const Move& move) {
  if (!has_forgone_plays(move)) return ForgonePlays::none();
  before.ensure_movegen_caches(dict);
  const TileCounts k = rack_tiles_of(move);
  TileCounts with_blank = k;
  with_blank.add(BLANK);
  BestByTau best;
  for (const Move& play : MoveGenerator(before, dict).generate(Rack::from_counts(with_blank))) {
    offer_blank_play(before, play, k, best);
  }
  return best.margins(move.score(), exhausted_after(before, k));
}

ForgonePlays compute_forgone_plays_reference(const Board& before, const Dictionary& dict,
                                             const Move& move) {
  if (!has_forgone_plays(move)) return ForgonePlays::none();
  const TileCounts k = rack_tiles_of(move);
  BestByTau best;
  MoveGenerator gen(before, dict);
  for (Tile tau = Tile::of(0); tau < TILE_KINDS; ++tau) {
    TileCounts rack = k;
    rack.add(tau);
    for (const Move& play : gen.generate(Rack::from_counts(rack))) {
      if (rack_tiles_of(play).count(tau) > k.count(tau)) best.offer(tau, play.score());
    }
  }
  return best.margins(move.score(), exhausted_after(before, k));
}

}  // namespace scribblez::belief
