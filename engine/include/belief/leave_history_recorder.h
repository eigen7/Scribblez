#pragma once

// The leave inference queries a player can make live, from what the game shows
// them: every move, both seats' (Agent::observe_move), and their own rack at
// each of their turns (Agent::make_move). It is the agent's mirror of what
// PositionEncoder's replay records from a .slog, so a query made in play is the
// training row of the same position: the opponent's last moves with the
// cross-checks before them, their forgone plays (computed when a query first
// needs them) and the leave-kind masks the observer's racks give them, and the observer's
// own last two exchanges. It never reads which tiles an opponent's exchange
// returned, which only the game knows.
//
// DeepBot keeps one: its own_query is Alice's belief about Bob's leave at the
// root, and own_moves feeds belief::ReplyView, Bob's view of her candidate.

#include "belief/forgone_plays.h"
#include "belief/leave_query.h"
#include "belief/reply_query.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"

#include <map>
#include <optional>
#include <vector>

namespace scribblez {

class Dictionary;
struct MoveRequest;

namespace belief {

class LeaveHistoryRecorder {
 public:
  // Forgets the game so far.
  void begin_game();

  // The game's next move, by either seat.
  void observe(const Move& move);

  // The query of the player to move with `req`, the recorder's owner: their
  // belief about the leave the opponent kept at their last move. Records the
  // rack, which the masks of the opponent's moves since read.
  LeaveQuery own_query(const MoveRequest& req);

  // The owner's last two moves before now, most recent first: what ReplyView
  // reads as the chooser's earlier moves.
  std::vector<ChooserMove> own_moves(const Dictionary& dict);

 private:
  struct Seen {
    Move move;
    Board before;
    int bag_before = 0;
    int mover_rack_before = 0;            // tiles on the mover's rack before it
    std::optional<ForgonePlays> forgone;  // computed on first use
  };

  const ForgonePlays& forgone(int k, const Dictionary& dict);
  // The history slot for the opponent's move `k`, its mask read from the
  // owner's rack at their next turn.
  LeaveHistorySlot opponent_slot(int k, const Dictionary& dict);
  LeaveHistory history(int turn, const Dictionary& dict);
  // The tiles the move at `k` returned to the bag, if it is the owner's
  // exchange; empty otherwise or before the game.
  Rack own_exchange(int k) const;

  std::vector<Seen> seen_;
  std::map<int, Rack> own_racks_;  // by turn
  Board board_;
  int bag_ = 0;
  std::array<int, 2> rack_sizes_{};
};

}  // namespace belief
}  // namespace scribblez
