#pragma once

#include "scribblez/agent.h"

#include <memory>
#include <string>
#include <vector>

namespace scribblez {

class WebSession;

// A human player driven through a WebSession: renders the position to the
// browser and blocks until the user submits a move (or passes / exchanges).
class HumanWebAgent : public Agent {
 public:
  HumanWebAgent(WebSession& session, const std::string& my_name, const std::string& opp_name);

  std::string name() const override { return my_name_; }
  Move make_move(const MoveRequest& req) override;

  // Build a HumanWebAgent from `--player "--type=human [options]"` tokens
  // (after the factory has stripped --type and --name). `session` is the
  // already-bound WebSession the browser will talk to. Throws on bad input.
  static std::unique_ptr<HumanWebAgent> from_spec(const std::vector<std::string>& tokens,
                                                  const std::string& name, WebSession& session,
                                                  const std::string& opp_name);

  // Human-readable description + options, shown by `play_game --help`.
  static std::string options_help();

 private:
  WebSession& session_;
  std::string my_name_;
  std::string opp_name_;
};

}  // namespace scribblez
