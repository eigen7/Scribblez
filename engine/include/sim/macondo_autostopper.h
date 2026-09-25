#pragma once

// The stopping rule of Macondo's simmer (montecarlo/stopping_condition.go) at
// the Stop99 setting BestBot uses: every kCheckInterval iterations it prunes
// the candidates that confidently trail the leader, and it ends the sim once
// one candidate remains or the iteration cap is reached.

#include "game/board.h"
#include "game/move.h"
#include "util/running_stat.h"

#include <cstdint>
#include <vector>

namespace scribblez {

// One simmed candidate's statistics, from the mover's point of view.
struct SimmedPlay {
  Move move;
  // Per-rollout win probability: an exact 0, 1/2 or 1 when the rollout
  // finished the game, else WinPctTable's estimate for its final position.
  util::RunningStat win_prob;
  // Per-rollout spread gained over the rollout, plus the value of the leaves
  // left by the last two plies (the mover's counted for, the opponent's
  // against).
  util::RunningStat equity;
  bool ignored = false;  // pruned; no longer simmed
};

// Sort `plays` best first: unpruned plays ahead of pruned ones, then by win
// probability, with differences within 1e-9 falling to equity. Stable, so
// exact ties keep the candidates' order.
void rank_simmed_plays(std::vector<SimmedPlay>& plays);

class MacondoAutoStopper {
 public:
  // Iterations between stopping checks.
  static constexpr int kCheckInterval = 128;

  // `board` is the decision point, which play-similarity checks read words
  // from.
  explicit MacondoAutoStopper(const Board& board) : board_(board) {}

  // Whether the sim should stop after `iterations` iterations of `plies`-ply
  // rollouts. May mark more plays ignored.
  bool should_stop(uint64_t iterations, std::vector<SimmedPlay>& plays, int plies) const;

 private:
  // Two plays that lay the same tiles along the same word extent, i.e. that
  // differ only in the order of their letters. Late in a sim, a trailing play
  // that is similar to the leader is pruned outright.
  bool materially_similar(const Move& a, const Move& b) const;

  const Board& board_;
};

}  // namespace scribblez
