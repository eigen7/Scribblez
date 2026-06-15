#include "scribblez/game.h"

#include "scribblez/movegen.h"

#include <cassert>
#include <utility>

namespace scribblez {

Game::Game(Agent& p0, Agent& p1, const Dictionary& dict, uint64_t seed)
    : dict_(dict), seed_(seed), bag_(seed) {
  players_[0] = &p0;
  players_[1] = &p1;
  log_.seed = seed;
  log_.player_names = {players_[0]->name(), players_[1]->name()};
}

void Game::set_initial_scores(std::array<int, 2> initial_scores) {
  assert(log_.turns.empty());  // must be set before play() begins
  scores_ = initial_scores;
  log_.initial_scores = initial_scores;
}

void Game::refill_rack(int p, Rack* drawn_out) {
  while (racks_[p].size() < RACK_SIZE) {
    auto t = bag_.draw();
    if (!t) break;
    racks_[p].add(*t);
    if (drawn_out) drawn_out->add(*t);
  }
}

void Game::play() {
  // Initial draws.
  for (int p = 0; p < 2; ++p) {
    refill_rack(p, /*drawn_out=*/nullptr);
  }
  log_.initial_racks[0] = racks_[0];
  log_.initial_racks[1] = racks_[1];

  int cur = 0;
  int consecutive_zero_turns = 0;
  constexpr int kMaxConsecutiveZero = 6;  // 3 per player
  constexpr int kMaxTurns = 400;          // safety net

  // One generator for the whole game: its cross-checks and anchors live on the
  // board and are maintained incrementally as moves are applied, so PASS/
  // EXCHANGE turns cost nothing and PLAY turns only touch the affected squares.
  MoveGenerator gen(board_, dict_);

  while ((int)log_.turns.size() < kMaxTurns) {
    // Generate legal plays for current player.
    std::vector<Move> legal = gen.generate(racks_[cur]);

    MoveRequest ctx{board_,           racks_[cur], racks_[1 - cur], scores_[cur],
                    scores_[1 - cur], bag_.size(), std::move(legal)};
    Move m = players_[cur]->make_move(ctx);

    TurnRecord rec;
    rec.player = cur;
    rec.rack_before = racks_[cur];
    rec.bag_size_before = bag_.size();
    rec.score_delta = 0;

    bool rack_emptied = false;
    const int n_glyphs = m.num_glyphs();
    if (m.type() == MoveType::PLAY) {
      // Remove placed tiles from rack, apply to board.
      for (int i = 0; i < n_glyphs; ++i) racks_[cur].remove(m.glyph(i).rack_tile());
      board_.apply(m);
      scores_[cur] += m.score();
      rec.score_delta = m.score();
      consecutive_zero_turns = 0;
      // Draw replacement tiles.
      refill_rack(cur, &rec.drawn);
      if (racks_[cur].empty() && bag_.size() == 0) {
        rack_emptied = true;
      }
    } else if (m.type() == MoveType::EXCHANGE) {
      // Remove tiles from rack, draw new ones, then put exchanged tiles back.
      for (int i = 0; i < n_glyphs; ++i) racks_[cur].remove(m.glyph(i).rack_tile());
      refill_rack(cur, &rec.drawn);
      for (int i = 0; i < n_glyphs; ++i) bag_.put_back(m.glyph(i).rack_tile());
      ++consecutive_zero_turns;
    } else {
      // PASS
      ++consecutive_zero_turns;
    }

    rec.move = std::move(m);
    rec.cumulative_scores = scores_;
    log_.turns.push_back(std::move(rec));

    if (rack_emptied) {
      // Standard end: out-going player gets sum of opponents' remaining tile values added,
      // opponents subtract their own remaining tile values.
      int opp = 1 - cur;
      int opp_remain = racks_[opp].point_value();
      scores_[cur] += opp_remain;
      scores_[opp] -= opp_remain;
      log_.end_reason = "out";
      break;
    }
    if (consecutive_zero_turns >= kMaxConsecutiveZero) {
      // Stalemate: each player subtracts their remaining tile values.
      for (int p = 0; p < 2; ++p) scores_[p] -= racks_[p].point_value();
      log_.end_reason = "stalemate";
      break;
    }
    cur = 1 - cur;
  }
  if (log_.end_reason.empty()) log_.end_reason = "max_turns";
  log_.final_scores = scores_;
  log_.final_racks = {racks_[0], racks_[1]};
}

}  // namespace scribblez
