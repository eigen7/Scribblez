#pragma once

// Real positions for tests: the decision point reached after some HastyBot vs
// HastyBot plies of a seeded game.

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "game/board.h"
#include "game/game.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace scribblez::testing {

// A decision point: the board, the mover's rack, and the scores from the
// mover's point of view.
struct Position {
  Board board;
  Rack rack;
  int my_score = 0;
  int opp_score = 0;
  int bag_size = 0;

  MoveRequest request(const Dictionary& dict) const {
    static const Rack kHidden;
    return MoveRequest{board, dict, rack, kHidden, my_score, opp_score, bag_size};
  }
};

// HastyBot, recording every decision point it is asked about into a log that
// both seats share, so one game yields all of its positions in ply order.
class RecordingHastyBot : public HastyBot {
 public:
  RecordingHastyBot(const char* name, std::vector<Position>& log)
      : HastyBot({.thread_id = 0, .name = name}), log_(log) {}
  MoveDecision make_move(const MoveRequest& req) override;

 private:
  std::vector<Position>& log_;
};

inline MoveDecision RecordingHastyBot::make_move(const MoveRequest& req) {
  log_.push_back({req.board, req.my_rack, req.my_score, req.opp_score, req.bag_size});
  return HastyBot::make_move(req);
}

// The decision points of the first `max_plies` plies of HastyBot-vs-HastyBot
// game `seed`: element k is the position after k moves, recorded only while
// the game is still going and the move before it was made with tiles in the
// bag.
inline std::vector<Position> hasty_positions(const Dictionary& dict, uint64_t seed, int max_plies) {
  std::vector<Position> log;
  RecordingHastyBot a0("A", log), a1("B", log);
  Game g(a0, a1, dict, seed);
  g.set_max_plies(max_plies);
  g.play();
  // Position k + 1 follows the move made from position k.
  for (size_t k = 0; k + 1 < log.size(); ++k) {
    if (log[k].bag_size == 0) {
      log.resize(k + 1);
      break;
    }
  }
  return log;
}

// The position after `plies` HastyBot-vs-HastyBot moves of game `seed`, or
// nullopt if the game did not stop there (it ended, or the bag was already
// empty).
inline std::optional<Position> hasty_position(const Dictionary& dict, uint64_t seed, int plies) {
  const std::vector<Position> log = hasty_positions(dict, seed, plies + 1);
  if (plies < 1 || plies >= int(log.size())) return std::nullopt;
  return log[plies];
}

// The first position of game `seed`, after at least one move and fewer than
// 40, whose bag size satisfies `want`.
inline std::optional<Position> find_position(const Dictionary& dict, uint64_t seed,
                                             const std::function<bool(int)>& want) {
  const std::vector<Position> log = hasty_positions(dict, seed, 40);
  for (size_t plies = 1; plies < log.size(); ++plies)
    if (want(log[plies].bag_size)) return log[plies];
  return std::nullopt;
}

}  // namespace scribblez::testing
