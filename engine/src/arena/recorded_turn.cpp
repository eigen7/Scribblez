#include "arena/recorded_turn.h"

#include "data/binary_log.h"
#include "game/bag.h"
#include "util/exception.h"

namespace scribblez {

namespace {

// The game's state before its first turn.
binlog::ReplayStart start_of(const GameLog& g) {
  return {.racks = g.initial_racks,
          .scores = g.initial_scores,
          .bag_size = Bag::kTotalTiles - g.initial_racks[0].size() - g.initial_racks[1].size(),
          .first_player = 0};
}

}  // namespace

RecordedPick pick_at_recorded_turn(Agent& agent, const GameLog& g, int turn,
                                   const Dictionary& dict) {
  if (turn < 0 || turn >= g.num_records)
    throw util::CleanException("turn {} is outside the game's {} turns", turn, g.num_records);
  const int seat = turn % 2;
  agent.begin_game({.initial_scores = g.initial_scores});
  binlog::ReplayStart state = start_of(g);
  Board board;
  const Rack hidden;
  for (int k = 0;; ++k) {
    const int mover = state.first_player;
    const Rack& opp_rack = state.bag_size == 0 ? state.racks[1 - mover] : hidden;
    const MoveRequest req{board,
                          dict,
                          state.racks[mover],
                          opp_rack,
                          state.scores[mover],
                          state.scores[1 - mover],
                          state.bag_size};
    if (k == turn) {
      return {.board = board,
              .rack = state.racks[mover],
              .score = state.scores[mover],
              .opp_score = state.scores[1 - mover],
              .bag = state.bag_size,
              .pick = agent.make_move(req).move};
    }
    if (mover == seat) agent.observe_turn(req);
    TurnRecord rec = g.records[k];
    state = binlog::replay_turn_records(state, &rec, 1);
    board.apply(rec.move);
    agent.observe_move(rec.move);
  }
}

}  // namespace scribblez
