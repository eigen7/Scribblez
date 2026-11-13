#pragma once

// DeepBot (docs/plans/deep_bot.md): Alice values each of her candidates by
// short sims on the GPU. Rollout i seats a leave for Bob drawn from her leave
// inference belief (belief::LeaveHistoryRecorder's query, the training row of
// her position) and draws the rest of both racks.
//
// Her candidates are the best N (N by the bag) of a screen: every legal move
// but the pass gets a few hasty sims, raced, as a policy-set pool labels a
// position, and the N likeliest best go on. Static equity buries the moves
// the sims exist to find (a setup, a fish, a one-tile play that keeps the bag
// open), so it does not choose the candidates. The pass stays out: where it
// sims best, the sims' opponent, who never passes back, is why.
//
// How Bob replies in the sims that then value the candidates is the reply
// model:
//
//   - `subtree`: Bob decides on his own information (agent/bob_subtree.h).
//     For rack i, his own sims give his moves their chances of being his
//     best, Alice's sims at the rack are apportioned among them by those
//     chances, and her payoffs after them, with her replenishment drawn,
//     value the candidate by their weighted mean. She plays the best.
//   - `hasty`: Bob and then Alice reply by HastyBot and the leaf network values
//     the end state, BestBot's loop with inferred racks and the leaf network.
//     The sims race the candidates in rounds (sim/pool_labeler.h), and she
//     plays the one most likely best.
//   - `alice-subtree`: Bob replies by HastyBot, and Alice's own sims on her
//     information choose her second move (agent/alice_subtree.h); two hasty
//     plies after it value the candidate. For a HastyBot opponent.
//
// Under the subtrees, a first pass of hasty sims over every candidate leaves
// its best for the reply model. A candidate that empties the bag has no
// subtree: Bob, holding every unseen tile, has nothing to infer. Under
// `subtree` the endgame solver makes his reply to it
// (agent/endgame_reply.h); under the other models it keeps the first pass's
// value, whose sims play the game to its end by HastyBot.
//
// Hidden leaves only, before the endgame: EndgameAgent<DeepBot> solves the turns
// with the bag empty, and DeepBot alone plays HastyBot's move there.

#include "agent/agent.h"
#include "agent/belief_truth.h"
#include "agent/bob_subtree.h"
#include "agent/policy_agreement.h"
#include "agent/policy_bot.h"
#include "agent/policy_scorer.h"
#include "agent/subtree_search.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "encoding/game_state_encoder.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/pool_labels.h"
#include "sim/rollout_values.h"
#include "sim/sim_runner.h"

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {

class Dictionary;

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
    int candidates = 10;       // Alice's, the screen's best
    int late_candidates = 32;  // ... with late_bag or fewer tiles in the bag
    int late_bag = 40;
    // The screen's rollouts per legal move, raced; 0: no screen, and the
    // candidates are the static-equity top moves.
    int screen_rollouts = 100;
    // The screen leaves out a blank move scoring this far below the
    // highest-scoring play, as the pools do (policy_set::pool_candidates).
    int blank_prune_points = 40;
    std::string reply = "subtree";  // the reply model: subtree, hasty or alice-subtree
    // hasty: rollouts per candidate, each with its own rack for Bob, raced.
    int rollouts = 1000;
    bool race = true;  // stop a clearly beaten candidate between rounds
    // The subtrees: R_A racks for Bob, each with a subtree and Alice's sims.
    int bob_racks = 100;
    // How many of Alice's candidates, the best by a first pass of hasty sims,
    // get the subtree; 0: all.
    int subtree_candidates = 8;
    // Alice's sims at each rack after Bob's reply; under `subtree`,
    // apportioned among his contenders.
    int sims_per_rack = 12;
    SubtreeParams subtree{};  // Bob's, under `subtree`
    // Under `subtree`, the node budget of each endgame solved after a
    // candidate that empties the bag; 0: such a candidate keeps the first
    // pass's value.
    uint64_t endgame_reply_budget = 2000;
    SubtreeParams second{};  // Alice's second turn's, under `alice-subtree`
    // A file to append a JSON line per turn to (agent/subtree_dump.h); costs
    // the hasty sims and a second run of the subtrees.
    std::string subtree_dump{};
    // The policy-set model (ONNX) the dump measures against Bob's replies
    // (agent/policy_agreement.h), served as PolicyBot serves it, on the same
    // leave inference model; needs the dump. Read by the factory, which needs
    // `dict` with it.
    std::string policy_model{};
    const Dictionary* dict = nullptr;
    uint64_t seed = 0;
  };

  // What the sims run on: the rollout backend, and the leave inference model's
  // forward pass with the leave table its beliefs are over (an empty `predict`
  // draws Bob's racks uniformly).
  struct Backend {
    std::shared_ptr<RolloutValues> values;
    belief::LeaveParamsFn predict;
    std::shared_ptr<const LeaveValueArrays> table;
    // The dump's policy-set model, with --policy-model.
    std::optional<PolicyBot::Backend> policy;
  };

  // Builds the backend Params names (deep_bot_factory.cpp, which reaches
  // CUDA and TensorRT).
  explicit DeepBot(const Params& params);
  DeepBot(const Params& params, Backend backend);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_turn(const MoveRequest& req) override;
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
  // How many candidates Alice values, by the bag.
  int candidate_count(const MoveRequest& req) const;
  // The moves her candidates are chosen from, best static equity first: the
  // screen's pool, or without a screen the candidates themselves.
  std::vector<Move> candidate_pool(const MoveRequest& req) const;
  // Her candidates: the screen's best of `pool`, or all of a pool no larger
  // than their count.
  std::vector<Move> candidates(const MoveRequest& req, std::vector<Move> pool,
                               const belief::LeaveSampler* belief) const;
  // Alice's belief about Bob's leave, null where the model has none to give.
  std::shared_ptr<const belief::LeaveSampler> root_belief(const belief::LeaveQuery& q) const;
  uint64_t turn_seed() const;
  // The hasty sims of `moves` over `rollouts` racks for Bob, dealt by `seed`,
  // and the rollouts they ran.
  struct HastySims {
    std::vector<PoolLabel> labels;
    long long sims = 0;
  };
  HastySims hasty_labels(const MoveRequest& req, const std::vector<Move>& moves,
                         const belief::LeaveSampler* belief, int rollouts, bool race,
                         uint64_t seed) const;
  Move subtree_move(const MoveRequest& req, const std::vector<Move>& moves,
                    const belief::LeaveSampler* belief);
  // Whether the endgame solver makes Bob's reply to a candidate that empties
  // the bag.
  bool solves_endgame_replies() const;
  // The candidates the hasty first pass (`hasty`) ranks among its best
  // subtree_candidates, in candidate order. One that empties the bag competes
  // only when the endgame solver will value it; otherwise all of those keep
  // their first-pass values, and the best are of the rest.
  std::vector<size_t> survivors(const std::vector<Move>& moves, const std::vector<PoolLabel>& hasty,
                                int bag) const;
  // Alice's value of each candidate that empties the bag, kPruned for the
  // rest: the endgame solver's for those of `kept`, or without it the first
  // pass's for all of them.
  std::vector<double> emptying_values(const MoveRequest& req, const AliceTurn& turn,
                                      const std::vector<PoolLabel>& hasty,
                                      const std::vector<size_t>& kept) const;
  // Alice's value of each candidate under the reply model's subtree, for the
  // `searched`, and `emptying`'s for the candidates it values; kPruned for
  // the rest.
  std::vector<double> bob_subtree_values(const MoveRequest& req, const AliceTurn& turn,
                                         const HastySims& hasty,
                                         const std::vector<size_t>& searched,
                                         const std::vector<double>& emptying);
  std::vector<double> second_turn_values(const MoveRequest& req, const AliceTurn& turn,
                                         const std::vector<size_t>& searched);
  void dump_turn(const MoveRequest& req, const AliceTurn& turn, const HastySims& hasty,
                 const std::vector<size_t>& searched, const BobReplies& replies,
                 const ReplyPayoffs& payoffs, const std::vector<double>& values);
  // The policy's agreement with Bob's replies after each searched candidate
  // (agent/policy_agreement.h); empty without a policy model.
  std::vector<PolicyAgreement> policy_agreements(const MoveRequest& req, const AliceTurn& turn,
                                                 const std::vector<size_t>& searched,
                                                 const BobReplies& replies);
  // What Bob's belief makes of the leave Alice keeps after each searched
  // candidate (agent/belief_truth.h); empty without a leave inference model.
  std::vector<BeliefTruth> belief_truths(const MoveRequest& req, const AliceTurn& turn,
                                         const std::vector<size_t>& searched) const;
  // What Alice's and her model of Bob's beliefs come from.
  BeliefModel belief_model() const { return {backend_.predict, backend_.table.get()}; }

  Params params_;
  Backend backend_;
  belief::LeaveHistoryRecorder recorder_;
  // The dump's policy scorer, and the encoder mirroring the live game (both
  // seats' moves) that gives Bob's position after a candidate; with
  // --policy-model only.
  std::unique_ptr<PolicyScorer> scorer_;
  std::optional<GameStateEncoder> encoder_;
  int ply_ = 0;    // moves observed this game
  int games_ = 0;  // games begun, this one included
};

// The `count` of `moves` that `labels` (parallel) make likeliest the best, in
// the order of `moves`; all of them when there are no more than `count`.
std::vector<Move> likeliest_best(const std::vector<Move>& moves,
                                 const std::vector<PoolLabel>& labels, int count);

}  // namespace scribblez
