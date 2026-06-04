#include "scribblez/player_factory.h"

#include "scribblez/agent.h"
#include "scribblez/web_server.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <stdexcept>

namespace scribblez {

std::string PlayerSpec::display_name() const {
  if (!name.empty()) return name;
  return type == PlayerType::Human ? "You" : "Greedy";
}

PlayerSpec parse_player_spec(const std::string& spec) {
  namespace po = boost::program_options;

  std::string type_str;
  PlayerSpec out;

  po::options_description desc("player options");
  desc.add_options()                                         //
    ("type", po::value<std::string>(&type_str)->required(),  //
     "player type: greedy | human")                          //
    ("name", po::value<std::string>(&out.name),              //
     "display name shown in the UI");

  try {
    // Each --player value is its own little option string; tokenize it the way
    // a shell would and feed it through program_options.
    std::vector<std::string> tokens = po::split_unix(spec);
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);  // enforces the required --type
  } catch (const std::exception& e) {
    throw std::runtime_error("bad --player spec \"" + spec + "\": " + e.what());
  }

  std::string t = boost::to_lower_copy(type_str);
  if (t == "greedy") {
    out.type = PlayerType::Greedy;
  } else if (t == "human") {
    out.type = PlayerType::Human;
  } else {
    throw std::runtime_error("bad --player spec \"" + spec + "\": unknown type '" + type_str +
                             "' (expected greedy or human)");
  }
  return out;
}

std::unique_ptr<Agent> make_player(const PlayerSpec& spec, WebSession* session,
                                   const std::string& opp_name) {
  switch (spec.type) {
    case PlayerType::Human:
      if (!session) {
        throw std::runtime_error("a human player needs a web session");
      }
      return std::make_unique<HumanWebAgent>(*session, spec.display_name(), opp_name);
    case PlayerType::Greedy:
      return std::make_unique<GreedyAgent>(spec.display_name());
  }
  throw std::runtime_error("unhandled player type");
}

}  // namespace scribblez
