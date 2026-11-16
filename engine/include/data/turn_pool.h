#pragma once

// What an agent's search of one turn leaves for the game's log, when the game
// asks for it (MoveRequest::record_pool): the turn's pool as the agent's
// screen labeled it, and what its second stage gave the screen's survivors.
// The game notes the turn, and the .slog writer puts each file's records into
// its .pool and .survivors sidecars (data/binary_log.h).

#include "data/pool_log.h"
#include "data/survivor_log.h"

namespace scribblez {

struct TurnPool {
  PoolPosition pool;
  TurnSurvivors survivors;
  PoolSources sources;        // of `pool`'s labels
  int survivor_rollouts = 0;  // the survivors' sims' budget
  int solved_racks = 0;       // the racks a solved survivor's value is the mean over
};

}  // namespace scribblez
