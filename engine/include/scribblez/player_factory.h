#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class WebSession;

// A parsed `--player` specification, e.g. `--player "--type=human --name=Dave"`.
// The factory extracts the universal options (--type and --name) and forwards
// everything else (`remaining_tokens`) to the chosen agent's static from_spec().
struct PlayerSpec {
  std::string type;                          // e.g. "greedy", "human", "hastybot" (lowercased)
  std::string name;                          // explicit --name, or empty
  std::vector<std::string> remaining_tokens;  // agent-specific tokens

  // Display name to show for this player (the explicit --name, else a default
  // based on the type, e.g. "You" for a human, "Greedy"/"HastyBot" for a bot).
  std::string display_name() const;

  // True iff this seat is a human player (needs a WebSession at construction).
  bool is_human() const;
};

// Parse one `--player` spec string (its own little option string, e.g.
// "--type=human" or "--type=greedy --name=Bot --seed=42") into a PlayerSpec.
// Throws std::runtime_error with a human-readable message on invalid input.
PlayerSpec parse_player_spec(const std::string& spec);

// Construct the agent for a parsed spec by dispatching to the chosen Agent
// subclass's from_spec(). A Human seat is driven through the browser, so
// `session` must be non-null when `spec.is_human()` (and is ignored otherwise).
// `opp_name` is the opponent's display name, shown in the UI for humans.
std::unique_ptr<Agent> make_player(const PlayerSpec& spec, WebSession* session,
                                   const std::string& opp_name);

// Concatenated help text for every registered agent type, formatted for
// `play_game --help`. Each block is the agent's options_help().
std::string all_player_types_help();

}  // namespace scribblez
