#pragma once

#include "scribblez/agent.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declared so Params::add_options() can register --player without
// pulling boost::program_options into every consumer of this header.
namespace boost::program_options { class options_description; }

namespace scribblez {

// A parsed `--player` specification, e.g. `--player "--type=human --name=Dave"`.
// Internal to PlayerFactory; exposed only because a few callers want the
// display name before agents are constructed.
struct PlayerSpec {
  std::string type;                          // e.g. "greedy", "human", "hastybot" (lowercased)
  std::string name;                          // explicit --name, or empty
  std::vector<std::string> remaining_tokens;  // agent-specific tokens

  // Display name to show for this player (the explicit --name, else a default
  // based on the type, e.g. "You" for a human, "Greedy"/"HastyBot" for a bot).
  std::string display_name() const;

  bool is_human() const;
};

// One-stop shop for the --player flow: registers the option, parses the raw
// strings into PlayerSpecs, and constructs the actual Agent instances. The
// CLI / app code never has to touch the per-spec dispatch loop directly.
class PlayerFactory {
 public:
  // Configuration knobs collected from the command line.
  struct Params {
    // Raw `--player` argument strings; populated by program_options via
    // add_options(). Empty means "default to two greedy players".
    std::vector<std::string> specs;

    // Register --player on the given options_description. Call from main()
    // before parsing argv, then pass the populated Params to make_players().
    void add_options(boost::program_options::options_description& desc);
  };

  // The two owned agents. Display names are reachable via agent->name().
  using Players = std::array<std::unique_ptr<Agent>, 2>;

  // Validate the raw `--player` specs in `params`, parse them, and construct
  // both agents. Defaults to two greedy players when no --player was given.
  // `thread_id` is the GameRunner thread index this pair will run on; it is
  // forwarded to each Agent's constructor (so agents can key per-thread
  // resources like MacondoOraclePool off it). Throws std::runtime_error with
  // a human-readable message on bad input.
  static Players make_players(const Params& params, int thread_id);

  // Concatenated help text for every registered agent type, formatted for
  // `play_game --help`.
  static std::string all_player_types_help();
};

}  // namespace scribblez
