#include "scribblez/game.h"

#include "scribblez/movegen.h"

#include <cassert>
#include <utility>

namespace scribblez {

Game::Game(std::unique_ptr<Agent> p0, std::unique_ptr<Agent> p1, const Dictionary& dict,
           uint64_t seed)
    : dict_(dict), seed_(seed), bag_(seed), rng_(seed ^ 0x9E3779B97F4A7C15ULL) {
  players_[0] = std::move(p0);
  players_[1] = std::move(p1);
  log_.seed = seed;
  log_.player_names = {players_[0]->name(), players_[1]->name()};
}

void Game::refill_rack(int p, std::vector<Tile>& drawn) {
  while (racks_[p].size() < RACK_SIZE) {
    auto t = bag_.draw();
    if (!t) break;
    racks_[p].add(*t);
    drawn.push_back(*t);
  }
}

void Game::play() {
  // Initial draws.
  for (int p = 0; p < 2; ++p) {
    std::vector<Tile> dummy;
    refill_rack(p, dummy);
  }

  int cur = 0;
  int consecutive_zero_turns = 0;
  constexpr int kMaxConsecutiveZero = 6;  // 3 per player
  constexpr int kMaxTurns = 400;          // safety net

  while ((int)log_.turns.size() < kMaxTurns) {
    // Generate legal plays for current player.
    MoveGenerator gen(board_, dict_);
    std::vector<Move> legal = gen.generate(racks_[cur]);

    AgentContext ctx{board_,           racks_[cur], scores_[cur],
                     scores_[1 - cur], bag_.size(), racks_[1 - cur].size(),
                     std::move(legal)};
    Move m = players_[cur]->choose(ctx, rng_);

    TurnRecord rec;
    rec.player = cur;
    rec.rack_before = racks_[cur];
    rec.bag_size_before = bag_.size();
    rec.score_delta = 0;

    bool rack_emptied = false;
    if (m.type == MoveType::PLAY) {
      // Remove placed tiles from rack, apply to board.
      for (const auto& t : m.tiles) {
        Tile rack_tile = t.is_blank ? BLANK : t.letter;
        bool ok = racks_[cur].remove(rack_tile);
        (void)ok;
        assert(ok);
      }
      board_.apply(m.tiles);
      scores_[cur] += m.score;
      rec.score_delta = m.score;
      consecutive_zero_turns = 0;
      // Draw replacement tiles.
      refill_rack(cur, rec.drawn);
      if (racks_[cur].empty() && bag_.size() == 0) {
        rack_emptied = true;
      }
    } else if (m.type == MoveType::EXCHANGE) {
      // Remove tiles from rack, draw new ones, then put exchanged tiles back.
      for (Tile L : m.exchanged) {
        bool ok = racks_[cur].remove(L);
        (void)ok;
        assert(ok);
      }
      refill_rack(cur, rec.drawn);
      for (Tile L : m.exchanged) bag_.put_back(L);
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
}

}  // namespace scribblez
