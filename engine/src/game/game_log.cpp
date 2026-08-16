#include "game/game_log.h"

namespace scribblez {

GameLog GameLogStorage::view() const {
  GameLog v;
  v.seed = seed;
  v.player_names = {player_names[0].c_str(), player_names[1].c_str()};
  v.initial_scores = initial_scores;
  v.initial_racks = initial_racks;
  v.records = turns.data();
  v.num_records = int(turns.size());
  v.final_scores = final_scores;
  v.final_racks = final_racks;
  v.end_reason = end_reason.c_str();
  v.num_random_opening_plies = num_random_opening_plies;
  return v;
}

}  // namespace scribblez
