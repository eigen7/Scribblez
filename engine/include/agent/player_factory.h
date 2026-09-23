#pragma once

#include "agent/agent.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace boost::program_options {
class options_description;
}

namespace scribblez {

// A parsed `--player` specification, e.g. `--player "--type=human --name=Dave"`.
struct PlayerSpec {
  std::string type;                           // e.g. "greedy", "human", "hastybot" (lowercased)
  std::string name;                           // explicit --name, or empty
  std::vector<std::string> remaining_tokens;  // agent-specific tokens

  // The explicit --name, else a default for the type ("You" for a human).
  std::string display_name() const;

  bool is_human() const;
};

// The --player flag: registers it, parses its values, and builds the agents.
class PlayerFactory {
 public:
  struct Params {
    std::vector<std::string> specs;  // raw --player strings; empty means greedy

    // Call before parsing argv.
    void add_options(boost::program_options::options_description& desc);
  };

  using Players = std::array<std::unique_ptr<Agent>, 2>;

  // Both seats' agents for one game thread. Defaults to two greedy players.
  // Throws util::CleanException on bad input.
  static Players make_players(const Params& params, int thread_id);

  static std::string all_player_types_help();
};

}  // namespace scribblez
