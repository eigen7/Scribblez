#include "scribblez/player_factory.h"

#include "scribblez/agent.h"
#include "scribblez/exception.h"
#include "scribblez/human_web_agent.h"
#include "scribblez/macondo_bot.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace scribblez {

namespace {

// Parse one --player spec string into a PlayerSpec. Implementation detail of
// PlayerFactory::make_players(); not part of the public API.
PlayerSpec parse_player_spec(const std::string& spec) {
  namespace po = boost::program_options;

  std::string type_str;
  PlayerSpec out;

  // Only the universal options (--type and --name) are parsed here. Anything
  // else is forwarded to the chosen agent's from_spec() as remaining tokens,
  // so adding a new agent never requires touching this function.
  po::options_description desc("player options");
  desc.add_options()                                         //
    ("type", po::value<std::string>(&type_str)->required(),  //
     "player type: greedy | human | hastybot")               //
    ("name", po::value<std::string>(&out.name),              //
     "display name shown in the UI");

  try {
    // Each --player value is its own little option string; tokenize it the way
    // a shell would and feed it through program_options with the rest passed
    // through unparsed.
    std::vector<std::string> tokens = po::split_unix(spec);
    po::parsed_options parsed =
      po::command_line_parser(tokens).options(desc).allow_unregistered().run();
    po::variables_map vm;
    po::store(parsed, vm);
    po::notify(vm);
    out.remaining_tokens = po::collect_unrecognized(parsed.options, po::include_positional);
  } catch (const std::exception& e) {
    throw std::runtime_error("bad --player spec \"" + spec + "\": " + e.what());
  }

  out.type = boost::to_lower_copy(type_str);
  if (out.type != "greedy" && out.type != "human" && out.type != "hastybot") {
    throw std::runtime_error("bad --player spec \"" + spec + "\": unknown type '" + type_str +
                             "' (expected greedy, human, or hastybot)");
  }
  return out;
}

// Dispatch one parsed spec to the chosen Agent subclass's from_spec().
std::unique_ptr<Agent> make_one(const PlayerSpec& spec, int thread_id,
                                const std::string& opp_name) {
  std::string name = spec.display_name();
  if (spec.type == "greedy") {
    return GreedyAgent::from_spec(spec.remaining_tokens, thread_id, name);
  }
  if (spec.type == "hastybot") {
    return HastyBotAgent::from_spec(spec.remaining_tokens, thread_id, name);
  }
  if (spec.type == "human") {
    return HumanWebAgent::from_spec(spec.remaining_tokens, thread_id, name, opp_name);
  }
  throw std::runtime_error("unhandled player type: " + spec.type);
}

}  // namespace

std::string PlayerSpec::display_name() const {
  if (!name.empty()) return name;
  if (type == "human") return "You";
  if (type == "hastybot") return "HastyBot";
  if (type == "greedy") return "Greedy";
  return type;  // unknown types: fall back to the literal type string
}

bool PlayerSpec::is_human() const { return type == "human"; }

void PlayerFactory::Params::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()("player", po::value<std::vector<std::string>>(&specs)->composing(),
                     "add a seat, e.g. --player \"--type=human\" --player \"--type=greedy\" "
                     "(repeat once per seat; default: two greedy)");
}

PlayerFactory::Players PlayerFactory::make_players(const Params& params, int thread_id) {
  try {
    std::vector<std::string> raw = params.specs;
    if (raw.empty()) raw = {"--type=greedy", "--type=greedy"};
    if (raw.size() != 2) {
      throw std::runtime_error("expected exactly two --player specs (got " +
                               std::to_string(raw.size()) + ")");
    }

    std::array<PlayerSpec, 2> specs;
    for (int s = 0; s < 2; ++s) specs[s] = parse_player_spec(raw[s]);

    // Build both agents. A Human agent's ctor blocks on its Vite dev server
    // coming up, so this is the point at which the browser UI appears.
    Players out;
    out[0] = make_one(specs[0], thread_id, specs[1].display_name());
    out[1] = make_one(specs[1], thread_id, specs[0].display_name());
    return out;
  } catch (const Exception&) {
    throw;  // already-printed user-facing error from a nested make_one()
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    throw Exception(e.what());
  }
}

std::string PlayerFactory::all_player_types_help() {
  std::ostringstream o;
  o << "--player \"--type=greedy [options]\"\n" << GreedyAgent::options_help() << "\n";
  o << "--player \"--type=hastybot [options]\"\n" << HastyBotAgent::options_help() << "\n";
  o << "--player \"--type=human [options]\"\n" << HumanWebAgent::options_help() << "\n";
  o << "Universal --player options (parsed by the factory before dispatch):\n"
    << "  --name=NAME   display name shown in the UI\n";
  return o.str();
}

}  // namespace scribblez
