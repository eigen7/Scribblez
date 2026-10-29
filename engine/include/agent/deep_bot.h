#pragma once

// DeepBot (docs/plans/deep_bot.md): Alice values each of her candidates (the
// static-equity top N, N by the bag) by short sims on the GPU. Rollout i seats
// a leave for Bob drawn from her leave inference belief
// (belief::LeaveHistoryRecorder's query, the training row of her position) and
// draws the rest of both racks. How Bob replies is the reply model:
//
//   - `subtree`: Bob decides on his own information (agent/bob_subtree.h).
//     For rack i, his own sims choose β, and Alice's sims after β, with her
//     replenishment drawn, value the candidate. She plays the best mean.
//   - `hasty`: Bob and then Alice reply by HastyBot and the leaf network values
//     the end state, BestBot's loop with inferred racks and the leaf network.
//     The sims race the candidates in rounds (sim/pool_labeler.h), and she
//     plays the one most likely best.
//
// Under either, a candidate that empties the bag gets the hasty sims, which
// then play the game to its end: Bob, holding every unseen tile, has nothing
// to infer.
//
// Hidden leaves only, before the endgame: EndgameAgent<DeepBot> solves the turns
// with the bag empty, and DeepBot alone plays HastyBot's move there.

#include "agent/agent.h"
#include "agent/bob_subtree.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/pool_labels.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class DeepBot : public Agent {
 public:
  static constexpr const char* kType = "deepbot";
  static constexpr bool kPreEndgameByDefault = false;

  struct Params {
    int thread_id = 0;
    std::string name;
    // The leaf network (a deep_bot_leaf_eval ONNX export), or "synthetic" for
    // the CPU reference with a logistic leaf (tests). Read by the factory.
    std::string leaf_model{};
    std::string leaf_precision = "fp16";
    // The leave inference model (ONNX) Bob's leaves are drawn from; empty draws
    // his whole rack uniformly. Read by the factory.
    std::string leave_inference{};
    int candidates = 10;       // Alice's, by static equity
    int late_candidates = 32;  // ... with late_bag or fewer tiles in the bag
    int late_bag = 40;
    std::string reply = "subtree";  // the reply model: subtree or hasty
    // hasty: rollouts per candidate, each with its own rack for Bob, raced.
    int rollouts = 1000;
    bool race = true;  // stop a clearly beaten candidate between rounds
    // subtree: R_A racks for Bob, each with Bob's subtree and Alice's sims.
    int bob_racks = 100;
    // How many of Alice's candidates, the best by a first pass of hasty sims,
    // get Bob's subtree; 0: all.
    int subtree_candidates = 8;
    int sims_per_rack = 4;
    SubtreeParams subtree{};
    // A file to append a JSON line per turn to (agent/subtree_dump.h); costs
    // the hasty sims and a second run of the subtrees.
    std::string subtree_dump{};
    uint64_t seed = 0;
  };

  // What the sims run on: the rollout backend, and the leave inference model's
  // forward pass with the leave table its beliefs are over (an empty `predict`
  // draws Bob's racks uniformly).
  struct Backend {
    std::shared_ptr<RolloutValues> values;
    belief::LeaveParamsFn predict;
    std::shared_ptr<const LeaveValueArrays> table;
  };

  // Builds the backend Params names (deep_bot_factory.cpp, which reaches
  // CUDA and TensorRT).
  explicit DeepBot(const Params& params);
  DeepBot(const Params& params, Backend backend);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=deepbot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<DeepBot> from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name);
  static std::string options_help();
  // Parse DeepBot's options, plus any a derived agent registered in `extra`.
  static Params parse_params(const std::vector<std::string>& tokens, int thread_id,
                             const std::string& name,
                             boost::program_options::options_description& extra,
                             const char* type_label);

 private:
  std::vector<Move> candidates(const MoveRequest& req) const;
  // Alice's belief about Bob's leave, null where the model has none to give.
  std::shared_ptr<const belief::LeaveSampler> root_belief(const belief::LeaveQuery& q) const;
  uint64_t turn_seed() const;
  // The hasty sims of `moves` over `rollouts` racks for Bob.
  std::vector<PoolLabel> hasty_labels(const MoveRequest& req, const std::vector<Move>& moves,
                                      const belief::LeaveSampler* belief, int rollouts,
                                      bool race) const;
  Move subtree_move(const MoveRequest& req, const std::vector<Move>& moves,
                    const belief::LeaveSampler* belief);
  // The candidates whose hasty first pass (`hasty`) ranks them among the best
  // subtree_candidates that do not empty the bag, in candidate order.
  std::vector<size_t> survivors(const std::vector<Move>& moves, const std::vector<PoolLabel>& hasty,
                                int bag) const;
  // Alice's value of each candidate: after Bob's subtree replies for the
  // `searched`, the first pass's for a candidate that empties the bag (its
  // hasty sims play the game to its end), and kPruned for the rest.
  std::vector<double> subtree_values(const MoveRequest& req, const AliceTurn& turn,
                                     const std::vector<PoolLabel>& hasty,
                                     const std::vector<size_t>& searched,
                                     const std::vector<std::vector<SubtreeMove>>& replies) const;
  void dump_turn(const MoveRequest& req, const AliceTurn& turn, const std::vector<PoolLabel>& hasty,
                 const std::vector<size_t>& searched,
                 const std::vector<std::vector<SubtreeMove>>& replies,
                 const std::vector<double>& values) const;
  BeliefModel bob_belief_model() const { return {backend_.predict, backend_.table.get()}; }

  Params params_;
  Backend backend_;
  belief::LeaveHistoryRecorder recorder_;
  int ply_ = 0;    // moves observed this game
  int games_ = 0;  // games begun, this one included
};

}  // namespace scribblez
