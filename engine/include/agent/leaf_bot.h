#pragma once

// LeafBot (--type=leafbot): the DeepBot leaf network
// (py/scribblez/deep_bot_leaf_eval) as a player, with no search; the harness
// that measures how good an evaluator the leaf is. Each turn it values each of
// HastyBot's top `candidates` moves by static equity at the position right
// after it: from its own side, the opponent to move, both leaves known
// (encoding/leaf_eval_input_encoder.h), which is the row the leaf trained on
// at every post-move turn of its games. The opponent's leave, which the game
// hides, takes `opp_leaves` values drawn from the leave inference model's
// belief at stratified quantiles (one draw where the leave is known to be
// empty), and a candidate's value is its leaf win value's mean over them;
// draws that land on the same leave are valued once, weighted by their
// count. It plays the best value, the earlier candidate (the better static
// equity) on ties, and its decision reports the candidates' values
// (MoveDecision::values). A candidate that empties the bag is valued like any
// other, since the leaf trains on every turn the game did not end at. Hidden
// leaves only, before the endgame: EndgameAgent<LeafBot> solves the turns
// with the bag empty.

#include "agent/agent.h"
#include "belief/leave_history_recorder.h"
#include "belief/leave_params.h"
#include "belief/leave_query.h"
#include "belief/leave_sampler.h"
#include "lexicon/leave_value_arrays.h"
#include "sim/rollout_values.h"

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace scribblez {

// One of the opponent's leaves a candidate is valued over, and its share of
// the value.
struct WeightedLeave {
  Rack leave;
  double weight = 1.0;
};

// The distinct leaves of `draws`, in order of first appearance, each weighted
// by its share of them.
std::vector<WeightedLeave> weighted_leaves(const std::vector<Rack>& draws);

class LeafBot : public Agent {
 public:
  static constexpr const char* kType = "leafbot";
  static constexpr bool kPreEndgameByDefault = false;

  struct Params {
    int thread_id = 0;
    std::string name;
    // The leaf network (a deep_bot_leaf_eval ONNX export), or "synthetic" for
    // the CPU reference's logistic leaf (tests). Read by the factory.
    std::string leaf_model{};
    std::string leaf_precision = "fp16";
    int leaf_rows = 4096;  // rows per leaf network call, as games' leaves are gathered
    // The leave inference model (ONNX) the opponent's leaves are drawn from.
    // Read by the factory, which requires it.
    std::string leave_inference{};
    int candidates = 10;  // HastyBot's best by static equity
    int opp_leaves = 16;  // the opponent's leaves each candidate is valued over
    uint64_t seed = 0;
  };

  // What LeafBot values candidates with: the leaf network (shared by every
  // game of the process, which gathers concurrent games' leaves into one
  // call), and the leave inference model's forward pass with the leave table
  // its beliefs are over. An empty `predict` draws the opponent's leaves from
  // the unseen tiles uniformly (tests).
  struct Backend {
    std::shared_ptr<RolloutValues> values;
    belief::LeaveParamsFn predict;
    std::shared_ptr<const LeaveValueArrays> table;
  };

  // Builds the backend Params names (leaf_bot_factory.cpp, which reaches CUDA
  // and TensorRT).
  explicit LeafBot(const Params& params);
  LeafBot(const Params& params, Backend backend);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_turn(const MoveRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=leafbot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<LeafBot> from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name);
  static std::string options_help();
  // Parse LeafBot's options, plus any a derived agent registered in `extra`.
  static Params parse_params(const std::vector<std::string>& tokens, int thread_id,
                             const std::string& name,
                             boost::program_options::options_description& extra,
                             const char* type_label);

  // Each candidate's value: its leaf win value's mean over the opponent's
  // leaves `opp`, weighted, from the side of the player who moves `moves`.
  // Public for the tests.
  std::vector<double> values(const MoveRequest& req, const std::vector<Move>& moves,
                             const std::vector<WeightedLeave>& opp) const;

 private:
  // `moves` and their values over this turn's opponent leaves (values()).
  std::vector<CandidateValue> valued(const MoveRequest& req, const std::vector<Move>& moves,
                                     const belief::LeaveQuery& q) const;
  // The opponent's leaves this turn's candidates are valued over: the
  // distinct draws, each weighted by its share of them.
  std::vector<WeightedLeave> opp_leaves(const belief::LeaveQuery& q) const;
  // The belief `q` asks for: the model's where it has one, else the unseen
  // tiles' prior over leaves of the query's length.
  std::shared_ptr<const belief::LeaveSampler> belief(const belief::LeaveQuery& q) const;
  uint64_t turn_seed() const;

  Params params_;
  Backend backend_;
  belief::LeaveHistoryRecorder recorder_;
  int ply_ = 0;         // moves observed this game
  uint64_t games_ = 0;  // games begun
};

// The index of the best of `values`, the earliest on ties.
size_t best_value(const std::vector<CandidateValue>& values);

}  // namespace scribblez
