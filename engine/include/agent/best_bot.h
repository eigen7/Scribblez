#pragma once

#include "agent/agent.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// A port of Macondo's BestBot (its SIMMING_BOT, ai/bot/elite.go) without its
// endgame engines, i.e. SIMMING_BOT_NO_EG; EndgameAgent<BestBot> adds the
// endgame solver. The phase follows the tiles unseen by the mover (bag plus
// opponent's rack):
//   - more than 14: sim HastyBot's top 40 moves by static equity, with
//     rollouts of max(min_sim_plies, 2) plies;
//   - 9 to 14: sim the top 80, with one rollout ply per unseen tile, which
//     usually reaches the end of the game;
//   - 8 or fewer (pre-endgame and endgame): HastyBot's move.
// The sim is MacondoSimmer, stopped by Macondo's Stop99 rule.
//
// Not ported: Woogles' wall-clock budget per move (a deterministic iteration
// cap, max_iterations, stands in for it), the multi-threaded default (sims run
// on sim_threads threads), and opponent-rack inference, which is a different
// Macondo bot (SIMMING_INFER_BOT). Tiles of the opponent's rack that the game
// shows (face-up leaves) seed every rollout, as Macondo's known-opponent-rack
// setting does.
class BestBot : public Agent {
 public:
  static constexpr const char* kType = "bestbot";

  struct Params {
    int thread_id = 0;
    std::string name;
    // Rollout plies while more than 14 tiles are unseen. Woogles runs 5.
    int min_sim_plies = 5;
    int sim_threads = 1;
    // Cap on sim iterations per move; 0 leaves it to the stopping rule.
    int max_iterations = 0;
    uint64_t seed = 0;
  };

  explicit BestBot(const Params& params);

  MoveDecision make_move(const MoveRequest& req) override;
  void begin_game(const BeginGameRequest& req) override;
  void observe_move(const Move& move) override;

  // Build from `--player "--type=bestbot [options]"` tokens, with --type and
  // --name already stripped. Throws on bad input.
  static std::unique_ptr<BestBot> from_spec(const std::vector<std::string>& tokens, int thread_id,
                                            const std::string& name);

  static std::string options_help();

  // Parse BestBot's options, plus any a derived agent registered in `extra`,
  // from `tokens` in one pass. `type_label` names the type in parse errors.
  static Params parse_params(const std::vector<std::string>& tokens, int thread_id,
                             const std::string& name,
                             boost::program_options::options_description& extra,
                             const char* type_label);

  // The sim seed on the turn after `ply` moves have been observed. Public so a
  // test can reproduce a decision's sim exactly.
  uint64_t sim_seed(int ply) const;

 private:
  int min_sim_plies_;
  int sim_threads_;
  int max_iterations_;
  uint64_t seed_;
  int ply_ = 0;              // moves observed this game, by either seat
  int scoreless_turns_ = 0;  // consecutive scoreless moves, which rollouts continue
};

}  // namespace scribblez
