#pragma once

// DeepBot's --subtree-dump: one JSON line per turn on how Bob's subtree
// replies (agent/bob_subtree.h) differ from HastyBot's, for each of Alice's
// candidates. It stands in for an offline subtree tool
// (docs/plans/deep_bot.md, build order step 4):
//
//   - how often β is Bob's equity-best move (HastyBot's reply), and the
//     equity ranks of the others;
//   - how sure Bob's sims were (the leader's q), and how many distinct replies
//     the racks drew;
//   - how many contenders his reply holds, at how many racks more than one,
//     and how far apart Alice's payoffs after them lie, which is where the
//     budget's stop rule reads whether his choice matters to her;
//   - the rollouts each stage of the turn ran;
//   - how often a second run of the subtrees, on fresh samples and draws, picks
//     the same reply: β's stability in R_B and the seed;
//   - Alice's value of each candidate under each reply model (null for the
//     subtree's where the hasty first pass pruned it), and each model's pick;
//   - per (rack, contender), the residual summary of Alice's sims after the
//     contender against the leaf network's prior there
//     (sim/map_residuals.h): what step 8's noise readout reads
//     (py/scripts/deep_bot/residual_noise.py).

#include "agent/bob_subtree.h"
#include "game/move.h"

#include <string>
#include <vector>

namespace scribblez {

// The rollouts a turn ran, by stage.
struct TurnSims {
  long long first_pass = 0;  // the hasty sims of every candidate
  long long subtrees = 0;    // Bob's
  long long after_replies = 0;
};

struct SubtreeTurn {
  // Which game: DeepBot `agent` on game thread `thread` plays its games in
  // turn, and `game` counts them, so lines from concurrent games sharing the
  // file group by (agent, thread, game).
  std::string agent;
  int thread = 0;
  int game = 0;
  int ply = 0;
  int bag = 0;
  const AliceTurn* turn = nullptr;
  std::vector<bool> searched;  // by candidate: Bob's subtree ran
  std::vector<double> subtree_values;
  std::vector<double> hasty_values;
  const std::vector<std::vector<BobReply>>* replies = nullptr;
  const std::vector<std::vector<BobReply>>* rerun = nullptr;  // the second seed's
  const ReplyPayoffs* payoffs = nullptr;
  TurnSims sims;
};

std::string subtree_turn_json(const SubtreeTurn& t);

// Appends `line` and a newline to `path`. Safe from concurrent game threads;
// throws util::CleanException when the write fails.
void append_dump_line(const std::string& path, const std::string& line);

}  // namespace scribblez
