#pragma once

// The opponent's view of one of the chooser's candidates
// (docs/plans/deep_bot.md, "Bob's subtree"): the leave inference query the
// opponent, holding a sampled rack, would make about the leave the candidate
// keeps. DeepBot's Alice makes one per (candidate, rack she samples for Bob)
// to model Bob's belief about her rack.
//
// It is the row training would encode at the opponent's turn with the
// observer's private history hidden (LeaveHistory::hide_private): inside the
// chooser's model, the opponent's past racks and exchanges are not the
// chooser's to see. So the opponent's exchanges are none, the older history
// slots' leave-kind masks read no rack, and the leave pool subtracts no
// exchange; the newest slot, the candidate itself, keeps a mask read from the
// sampled rack and that pool. Training hides it in a fraction of rows, and the
// row's indicator tells the model which reading it is in.

#include "belief/forgone_plays.h"
#include "belief/leave_query.h"
#include "game/board.h"
#include "game/game_log.h"
#include "game/move.h"
#include "game/rack.h"

#include <span>
#include <vector>

namespace scribblez {

class Dictionary;

namespace belief {

// One of the chooser's moves before the candidate, as it was played.
struct ChooserMove {
  Move move;
  Board board_before;
  int bag_before = 0;
  ForgonePlays forgone = ForgonePlays::none();
};

// The position the chooser decides at.
struct ReplyPosition {
  const Board& board;  // before the candidate, natural frame
  const Rack& chooser_rack;
  int chooser_score = 0;
  int opp_score = 0;
  int bag = 0;
  // The chooser's last two moves, most recent first; fewer early in the game.
  std::span<const ChooserMove> earlier;
};

// The opponent's view after the chooser plays `candidate` and draws, built once
// per candidate (the board, the history and the candidate's forgone plays) and
// queried once per rack sampled for the opponent.
class ReplyView {
 public:
  ReplyView(const ReplyPosition& pos, const Move& candidate, const Dictionary& dict);

  // The query of the opponent holding `opp_rack`.
  LeaveQuery query(const Rack& opp_rack) const;

 private:
  const Dictionary& dict_;
  Board after_;           // with its move-generation caches
  LeaveHistory history_;  // the candidate's slot mask is the rack's, set per query
  int length_;
  LeaveRowCause cause_;
  int bag_;
  int score_diff_;  // the opponent's lead after the candidate
};

// The last two moves of turn `turn`'s mover in `g`, most recent first, their
// forgone plays read from `forgone` (the game's .forgone entries, by turn).
std::vector<ChooserMove> chooser_moves_from_replay(const GameLog& g, int turn,
                                                   std::span<const ForgonePlays> forgone,
                                                   const Dictionary& dict);

}  // namespace belief
}  // namespace scribblez
