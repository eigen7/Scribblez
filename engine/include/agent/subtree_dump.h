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
//   - how often a second run of the subtrees, on fresh samples and draws, picks
//     the same reply: β's stability in R_B and the seed;
//   - Alice's value of each candidate under each reply model (null for the
//     subtree's where the hasty first pass pruned it), and each model's pick.

#include "agent/bob_subtree.h"
#include "game/move.h"

#include <string>
#include <vector>

namespace scribblez {

struct SubtreeTurn {
  int ply = 0;
  int bag = 0;
  const AliceTurn* turn = nullptr;
  std::vector<bool> searched;  // by candidate: Bob's subtree ran
  std::vector<double> subtree_values;
  std::vector<double> hasty_values;
  const std::vector<std::vector<BobReply>>* replies = nullptr;
  const std::vector<std::vector<BobReply>>* rerun = nullptr;  // the second seed's
};

std::string subtree_turn_json(const SubtreeTurn& t);

// Appends `line` and a newline to `path`. Safe from concurrent game threads.
void append_dump_line(const std::string& path, const std::string& line);

}  // namespace scribblez
