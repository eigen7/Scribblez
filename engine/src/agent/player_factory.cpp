#include "agent/player_factory.h"

#include "agent/agent.h"
#include "agent/endgame_hasty_bot.h"
#include "agent/greedy_agent.h"
#include "agent/human_web_agent.h"
#include "agent/macondo_bot.h"
#include "agent/mset_sim_agent.h"
#include "agent/neural_agent.h"
#include "agent/neural_sim_agent.h"
#include "agent/sim_agent.h"
#include "agent/ultimate_bot_agent.h"
#include "agent/weird_bot.h"
#include "util/exception.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <array>
#include <sstream>
#include <string_view>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Everything the factory needs to know about a player type lives in one row of
// kPlayerTypes below, so adding a type is a single new entry -- no per-type
// switch scattered across parse_player_spec(), make_one(), display_name(), and
// all_player_types_help(); each of those iterates the list generically.
struct PlayerType {
  std::string_view type_str;      // --type value (lowercased)
  std::string_view default_name;  // display name when --name is omitted
  std::string (*options_help)();  // the agent's per-type --player help block
  // Construct the agent. opp_name is the other seat's display name; only the
  // human seat's from_spec takes it, so the uniform adapter ignores it.
  std::unique_ptr<Agent> (*build)(const std::vector<std::string>& tokens, int thread_id,
                                  const std::string& name, const std::string& opp_name);
};

// Adapter for the nine agents whose from_spec is the uniform three-arg form:
// drop opp_name, forward the rest. from_spec is a non-type template argument,
// so one template covers them all and each returns its own unique_ptr subtype
// (implicitly converted to unique_ptr<Agent>).
template <auto FromSpec>
std::unique_ptr<Agent> build_agent(const std::vector<std::string>& tokens, int thread_id,
                                   const std::string& name, const std::string& /*opp_name*/) {
  return FromSpec(tokens, thread_id, name);
}

// The human seat is the lone outlier: its from_spec also takes the opponent's
// display name (shown in the browser UI).
std::unique_ptr<Agent> build_human(const std::vector<std::string>& tokens, int thread_id,
                                   const std::string& name, const std::string& opp_name) {
  return HumanWebAgent::from_spec(tokens, thread_id, name, opp_name);
}

// The single source of per-type knowledge. Adding a player type is one new row.
constexpr std::array<PlayerType, 10> kPlayerTypes{{
  {"greedy", "Greedy", &GreedyAgent::options_help, &build_agent<&GreedyAgent::from_spec>},
  {"human", "You", &HumanWebAgent::options_help, &build_human},
  {"hastybot", "HastyBot", &HastyBotAgent::options_help, &build_agent<&HastyBotAgent::from_spec>},
  {"hastybot-endgame", "EndgameHastyBot", &EndgameHastyBotAgent::options_help,
   &build_agent<&EndgameHastyBotAgent::from_spec>},
  {"mset-sim", "MsetSim", &MsetSimAgent::options_help, &build_agent<&MsetSimAgent::from_spec>},
  {"neural", "Neural", &NeuralAgent::options_help, &build_agent<&NeuralAgent::from_spec>},
  {"neural-sim", "NeuralSim", &NeuralSimAgent::options_help,
   &build_agent<&NeuralSimAgent::from_spec>},
  {"sim", "SimBot", &SimAgent::options_help, &build_agent<&SimAgent::from_spec>},
  {"ultimatebot", "UltimateBot", &UltimateBotAgent::options_help,
   &build_agent<&UltimateBotAgent::from_spec>},
  {"weirdbot", "WeirdBot", &WeirdBotAgent::options_help, &build_agent<&WeirdBotAgent::from_spec>},
}};

// The entry whose type_str matches `type` (already lowercased), or nullptr.
const PlayerType* find_player_type(std::string_view type) {
  for (const PlayerType& pt : kPlayerTypes) {
    if (pt.type_str == type) return &pt;
  }
  return nullptr;
}

// "greedy | human | ... | weirdbot" for the --type help line.
std::string type_choices_bar() {
  std::string out;
  for (const PlayerType& pt : kPlayerTypes) {
    if (!out.empty()) out += " | ";
    out += pt.type_str;
  }
  return out;
}

// "greedy, human, ..., or weirdbot" for the unknown-type error message.
std::string type_choices_prose() {
  std::string out;
  for (std::size_t i = 0; i < kPlayerTypes.size(); ++i) {
    if (i > 0) out += ", ";
    if (i + 1 == kPlayerTypes.size()) out += "or ";
    out += kPlayerTypes[i].type_str;
  }
  return out;
}

// The --player options common to every agent type. Built by both
// parse_player_spec() and all_player_types_help(), so the parsed options and
// the documented ones share one source of truth.
po::options_description universal_player_options(std::string& type_str, std::string& name) {
  static const std::string type_help = "player type: " + type_choices_bar();
  po::options_description desc;
  desc.add_options()                                         //
    ("type", po::value<std::string>(&type_str)->required(),  //
     type_help.c_str())                                      //
    ("name", po::value<std::string>(&name),                  //
     "display name shown in the UI");
  return desc;
}

// Parse one --player spec string. An implementation detail of
// PlayerFactory::make_players().
PlayerSpec parse_player_spec(const std::string& spec) {
  std::string type_str;
  PlayerSpec out;

  // Only the universal options (--type and --name) are parsed here. Anything
  // else is forwarded to the chosen agent's from_spec() as remaining tokens,
  // so adding a new agent never requires touching this function.
  po::options_description desc = universal_player_options(type_str, out.name);

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
    throw util::CleanException("bad --player spec \"{}\": {}", spec, e.what());
  }

  out.type = boost::to_lower_copy(type_str);
  if (find_player_type(out.type) == nullptr) {
    throw util::CleanException("bad --player spec \"{}\": unknown type '{}' (expected {})", spec,
                               type_str, type_choices_prose());
  }
  return out;
}

// Dispatch to the chosen Agent subclass's from_spec() via its table entry.
std::unique_ptr<Agent> make_one(const PlayerSpec& spec, int thread_id,
                                const std::string& opp_name) {
  const PlayerType* pt = find_player_type(spec.type);
  if (pt == nullptr) throw util::Exception("unhandled player type: {}", spec.type);
  return pt->build(spec.remaining_tokens, thread_id, spec.display_name(), opp_name);
}

}  // namespace

std::string PlayerSpec::display_name() const {
  if (!name.empty()) return name;
  const PlayerType* pt = find_player_type(type);
  if (pt != nullptr) return std::string(pt->default_name);
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
  std::vector<std::string> raw = params.specs;
  if (raw.empty()) raw = {"--type=greedy", "--type=greedy"};
  if (raw.size() != 2) {
    throw util::CleanException("expected exactly two --player specs (got {})", raw.size());
  }

  std::array<PlayerSpec, 2> specs;
  for (int s = 0; s < 2; ++s) specs[s] = parse_player_spec(raw[s]);

  // Build both agents. A Human agent's ctor blocks on its Vite dev server
  // coming up, so this is the point at which the browser UI appears.
  Players out;
  out[0] = make_one(specs[0], thread_id, specs[1].display_name());
  out[1] = make_one(specs[1], thread_id, specs[0].display_name());
  return out;
}

std::string PlayerFactory::all_player_types_help() {
  std::ostringstream o;
  for (const PlayerType& pt : kPlayerTypes) {
    o << "--player \"--type=" << pt.type_str << " [options]\"\n" << pt.options_help() << "\n";
  }
  std::string type_str, name;  // scratch binding targets; never read here
  o << "Universal --player options (parsed by the factory before dispatch):\n"
    << universal_player_options(type_str, name);
  return o.str();
}

}  // namespace scribblez
