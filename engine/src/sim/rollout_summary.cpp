#include "sim/rollout_summary.h"

#include "game/board.h"

#include <algorithm>
#include <bitset>
#include <cmath>

namespace scribblez {

namespace {

using SquareSet = std::bitset<BOARD_SIZE * BOARD_SIZE>;

// The squares orthogonally adjacent to a tile `m` places.
SquareSet neighbor_squares(const Move& m) {
  SquareSet out;
  if (m.type() != MoveType::PLAY) return out;
  visit_placed_squares(m, [&](int r, int c) {
    if (r > 0) out.set(size_t((r - 1) * BOARD_SIZE + c));
    if (r + 1 < BOARD_SIZE) out.set(size_t((r + 1) * BOARD_SIZE + c));
    if (c > 0) out.set(size_t(r * BOARD_SIZE + c - 1));
    if (c + 1 < BOARD_SIZE) out.set(size_t(r * BOARD_SIZE + c + 1));
  });
  return out;
}

bool places_tile_in(const Move& m, const SquareSet& squares) {
  bool hit = false;
  visit_placed_squares(m, [&](int r, int c) { hit = hit || squares[size_t(r * BOARD_SIZE + c)]; });
  return hit;
}

void add_next_move(const Move& m, const SquareSet& beside_candidate, NextMoveStats* stats) {
  if (m.type() != MoveType::PLAY) {
    ++stats->non_plays;
    ++stats->score_hist[0];
    return;
  }
  stats->score_sum += m.score();
  ++stats->score_hist[size_t(std::min(m.score() / kScoreBinWidth, kScoreBins - 1))];
  stats->bingos += m.num_glyphs() == RACK_SIZE;
  stats->adjacent += places_tile_in(m, beside_candidate);
}

int delta_bin(double delta) {
  const int bin = int(std::floor((delta - kDeltaBinFloor) / kDeltaBinWidth)) + 1;
  return std::clamp(bin, 0, kDeltaBins - 1);
}

int end_swing_bin(int swing) {
  const int bin = int(std::floor(double(swing - kEndSwingBinFloor) / kEndSwingBinWidth)) + 1;
  return std::clamp(bin, 0, kEndSwingBins - 1);
}

void add_game_end(const RolloutResult& r, RolloutSummary* s) {
  s->self_stranded_sum += r.self_stranded;
  s->opp_stranded_sum += r.opp_stranded;
  s->self_went_out += r.self_stranded == 0;
  s->opp_went_out += r.self_stranded != 0 && r.opp_stranded == 0;
  const int swing = end_rack_swing(r);
  s->end_swing_sum += swing;
  ++s->end_swing_hist[size_t(end_swing_bin(swing))];
}

}  // namespace

PairedWinDiff paired_win_diff(std::span<const RolloutResult> a, std::span<const RolloutResult> b) {
  PairedWinDiff out;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = (a[i].p_win + 0.5 * a[i].p_draw) - (b[i].p_win + 0.5 * b[i].p_draw);
    out.sum += d;
    out.sq_sum += d * d;
  }
  return out;
}

bool clearly_below(const PairedWinDiff& d, size_t n, double sigmas) {
  const double mean = d.sum / double(n);
  const double se = std::sqrt(std::max(d.sq_sum / double(n) - mean * mean, 0.0) / double(n));
  return mean + sigmas * se < 0;
}

RolloutSummary summarize_rollouts(const Move& candidate, std::span<const RolloutResult> rollouts) {
  const SquareSet beside = neighbor_squares(candidate);
  RolloutSummary s;
  for (const RolloutResult& r : rollouts) {
    ++s.n;
    s.wins += r.p_win;
    s.draws += r.p_draw;
    s.losses += r.p_loss;
    s.delta_sum += r.delta;
    s.delta_sq_sum += r.delta_sq;
    ++s.delta_hist[size_t(delta_bin(r.delta))];
    add_next_move(r.opp_reply, beside, &s.opp_reply);
    add_next_move(r.self_next, beside, &s.self_next);
    add_game_end(r, &s);
  }
  return s;
}

}  // namespace scribblez
