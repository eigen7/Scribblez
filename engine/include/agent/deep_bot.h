#pragma once

// DeepBot (docs/plans/deep_bot.md), with the `hasty` reply model: Alice values
// each of her candidates (the static-equity top N, N by the bag) by short sims
// on the GPU. Rollout i seats a leave for Bob drawn from her leave inference
// belief (belief::LeaveHistoryRecorder's query, the training row of her
// position) and draws the rest of both racks; Bob and then Alice reply by
// HastyBot, and the leaf network values the end state. The sims race the
// candidates in rounds (sim/pool_labeler.h), and she plays the one most likely
// best. This is BestBot's loop with inferred racks and the leaf network; the
// `subtree` reply model, Bob deciding on his own information, comes next.
//
// Hidden leaves only, before the endgame: EndgameAgent<DeepBot> solves the turns
// with the bag empty, and DeepBot alone plays HastyBot's move there.

#include "agent/agent.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/rollout_values.h"

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
    int rollouts = 1000;  // per candidate, each with its own rack for Bob
    bool race = true;     // stop a clearly beaten candidate between rounds
    int threads = 0;      // CPU threads (the samplers, the cpu backend); 0: all
    uint64_t seed = 0;
  };

  // What the sims run on: the rollout backend, and the leave inference model's
  // forward pass with the leave table its beliefs are over (an empty `predict`
  // draws Bob's racks uniformly).
  struct Backend {
    std::unique_ptr<RolloutValues> values;
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

  Params params_;
  Backend backend_;
  belief::LeaveHistoryRecorder recorder_;
  int ply_ = 0;  // moves observed this game
};

}  // namespace scribblez
