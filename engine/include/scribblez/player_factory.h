#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>

namespace scribblez {

class WebSession;

// The kind of agent a seat is played by.
enum class PlayerType { Greedy, Human };

// A parsed `--player` specification, e.g. `--player "--type=human --name=Dave"`.
struct PlayerSpec {
  PlayerType type = PlayerType::Greedy;
  std::string name;  // explicit display name, or empty to use the default

  // Display name to show for this player (the explicit --name, else a default
  // based on the type: "You" for a human, "Greedy" for the greedy agent).
  std::string display_name() const;
};

// Parse one `--player` spec string (its own little option string, e.g.
// "--type=human" or "--type=greedy --name=Bot") into a PlayerSpec. Throws
// std::runtime_error with a human-readable message on invalid input.
PlayerSpec parse_player_spec(const std::string& spec);

// Construct the agent for a parsed spec. A Human seat is driven through the
// browser, so `session` must be non-null for PlayerType::Human (and is ignored
// otherwise). `opp_name` is the opponent's display name, shown in the UI.
std::unique_ptr<Agent> make_player(const PlayerSpec& spec, WebSession* session,
                                   const std::string& opp_name);

}  // namespace scribblez
