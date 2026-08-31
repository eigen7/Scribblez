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

namespace nn {
class ServiceCache;
}

// A parsed `--player` specification, e.g. `--player "--type=human --name=Dave"`.
struct PlayerSpec {
  std::string type;                           // e.g. "greedy", "human", "hastybot" (lowercased)
  std::string name;                           // explicit --name, or empty
  std::vector<std::string> remaining_tokens;  // agent-specific tokens

  // The explicit --name, else a default for the type ("You" for a human).
  std::string display_name() const;

  bool is_human() const;
};

// The whole --player flow: registers the option, parses the raw strings, and
// constructs the agents.
class PlayerFactory {
 public:
  struct Params {
    std::vector<std::string> specs;  // raw --player strings; empty means greedy

    // Call before parsing argv.
    void add_options(boost::program_options::options_description& desc);
  };

  using Players = std::array<std::unique_ptr<Agent>, 2>;

  // Validate and parse the raw `--player` specs and return both agents.
  // Defaults to two greedy players. Throws util::CleanException on bad input.
  // Model-driven seats resolve their (run-shared) service through `services`,
  // so calling this once per thread yields agents that share one loaded model
  // per distinct spec rather than one apiece.
  static Players make_players(const Params& params, int thread_id, nn::ServiceCache& services);

  static std::string all_player_types_help();
};

}  // namespace scribblez
