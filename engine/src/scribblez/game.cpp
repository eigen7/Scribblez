#include "scribblez/game.h"

#include <cassert>
#include <utility>

namespace scribblez {

GameLog GameLogStorage::view() const {
  GameLog v;
  v.seed = seed;
  v.player_names = {player_names[0].c_str(), player_names[1].c_str()};
  v.initial_scores = initial_scores;
  v.initial_racks = initial_racks;
  v.records = turns.data();
  v.num_records = static_cast<int>(turns.size());
  v.final_scores = final_scores;
  v.final_racks = final_racks;
  v.end_reason = end_reason.c_str();
  return v;
}

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

  // Let stateful agents reset to a clean position (seats alternate across a
  // series, so the same Agent instances are reused game to game).
  players_[0]->begin_game(scores_);
  players_[1]->begin_game(scores_);

  int cur = 0;
  int consecutive_zero_turns = 0;
  constexpr int kMaxConsecutiveZero = 6;  // 3 per player
  constexpr int kMaxTurns = 400;          // safety net

  while ((int)log_.turns.size() < kMaxTurns) {
    // The game loop no longer generates moves; each agent generates the moves
    // it needs from the board + dictionary on its own turn. The board's
    // cross-check/anchor caches still live on board_ and are maintained
    // incrementally as moves are applied.
    MoveRequest ctx{board_,           dict_,      racks_[cur], racks_[1 - cur], scores_[cur],
                    scores_[1 - cur], bag_.size()};
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

    // Notify both seats of the applied move, in turn order, so stateful agents
    // (e.g. NeuralAgent) can mirror the full game even on the opponent's
    // turns. The board/score are already updated above for a PLAY.
    players_[0]->observe_move(rec.move);
    players_[1]->observe_move(rec.move);

    rec.cumulative_scores = scores_;
    log_.turns.push_back(std::move(rec));

    if (rack_emptied) {
      // Standard end: the out-going player gains twice the sum of the
      // opponents' remaining tile values, and the opponents' scores are left
      // unchanged. This is the modern tournament convention (a single +2N
      // bonus) rather than awarding +N to the winner and -N to the loser.
      int opp = 1 - cur;
      int opp_remain = racks_[opp].point_value();
      scores_[cur] += 2 * opp_remain;
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
