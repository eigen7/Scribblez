// position_pick_tool: asks an agent for its move at positions of recorded
// games, each replayed to as the agent would have lived it
// (arena/recorded_turn.h). It is the tool for asking "what does this agent
// play here?" of positions whose answer something else already knows: the
// blind spots (docs/blind_spots.md), through
// py/scripts/deep_bot/blind_spot_picks.py.
//
//   position_pick_tool --positions list.txt --out picks.jsonl
//     --player "--type=deepbot-endgame --leaf-model leaf.onnx --reply hasty"
//
// --positions names a file of lines "<.slog path> <game> <turn>", both
// 0-based; the agent takes the seat that moved at the turn. --out gets a JSON
// line per position, in the file's order: its "slog", "game" and "turn", the
// mover's "rack" and the "bag" there, the "pick" in the notation of the
// reports (data/gcg_writer.h's move_notation), and the "seconds" the agent
// took. One agent plays every position, one at a time, on the process's
// threads.

#include "agent/player_factory.h"
#include "arena/recorded_turn.h"
#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "game/game_log.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/io.h"
#include "util/misc.h"
#include "util/seed_producer.h"
#include "util/work_pool.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace scribblez {
namespace {

namespace json = boost::json;

struct Options {
  std::string positions;
  std::string player;
  std::string out;
  int threads = util::default_thread_count();
};

// One line of --positions.
struct Asked {
  std::string slog;
  uint32_t game = 0;
  int turn = 0;
};

void add_options(boost::program_options::options_description& desc, Options& opt) {
  namespace po = boost::program_options;
  desc.add_options()("help,h", "show this help message and exit")(
    "positions", po::value<std::string>(&opt.positions),
    "a file of lines \"<.slog path> <game> <turn>\" (0-based)")(
    "player", po::value<std::string>(&opt.player),
    "the agent, as play_game's --player names it: \"--type=X [options]\"")(
    "out", po::value<std::string>(&opt.out), "the JSON-lines file to write (default: stdout)")(
    "threads", po::value<int>(&opt.threads)->default_value(opt.threads),
    "CPU threads, the process's budget for the agent's parallel work");
}

void validate(const Options& opt) {
  if (opt.positions.empty()) throw util::CleanException("--positions is required");
  if (opt.player.empty()) throw util::CleanException("--player is required");
  if (opt.threads < 1) throw util::CleanException("--threads must be >= 1");
}

std::vector<Asked> read_positions(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw util::CleanException("cannot open {}", path);
  std::vector<Asked> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream fields(line);
    Asked a;
    if (!(fields >> a.slog >> a.game >> a.turn) || a.turn < 0)
      throw util::CleanException("{}: expected \"<.slog path> <game> <turn>\", got \"{}\"", path,
                                 line);
    out.push_back(std::move(a));
  }
  return out;
}

// The .slog files the positions name, each loaded once and its header checked.
std::map<std::string, std::vector<char>> load_slogs(const std::vector<Asked>& asked) {
  std::map<std::string, std::vector<char>> out;
  for (const Asked& a : asked) {
    if (out.contains(a.slog)) continue;
    std::vector<char> buf = util::read_file_bytes(a.slog);
    const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
    if (buf.size() < sizeof(binlog::FileHeader) || hdr->magic != binlog::kMagic ||
        hdr->version != binlog::kVersion)
      throw util::CleanException("{}: not a version-{} .slog", a.slog, binlog::kVersion);
    if ((hdr->flags & binlog::kFlagFaceUpLeaves) != 0)
      throw util::CleanException("{} was played with face-up leaves; the replay hides them",
                                 a.slog);
    out.emplace(a.slog, std::move(buf));
  }
  return out;
}

json::object pick_json(const Asked& a, const RecordedPick& p, double seconds) {
  return {{"slog", a.slog},    {"game", a.game},
          {"turn", a.turn},    {"rack", p.rack.to_string()},
          {"bag", p.bag},      {"pick", move_notation(p.board, p.pick)},
          {"seconds", seconds}};
}

// `agent`'s pick at `a`, a position of the loaded .slog `buf`, as its line.
json::object ask(Agent& agent, const Asked& a, const std::vector<char>& buf,
                 const Dictionary& dict) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  if (a.game >= hdr->num_games)
    throw util::CleanException("{}: game {} of {}", a.slog, a.game, hdr->num_games);
  std::vector<TurnRecord> scratch;
  const GameLog g = binlog::make_game_view(buf.data(), a.game, scratch, nullptr);
  const auto began = std::chrono::steady_clock::now();
  const RecordedPick pick = pick_at_recorded_turn(agent, g, a.turn, dict);
  const std::chrono::duration<double> took = std::chrono::steady_clock::now() - began;
  return pick_json(a, pick, took.count());
}

void run(const Options& opt) {
  const std::vector<Asked> asked = read_positions(opt.positions);
  const std::map<std::string, std::vector<char>> slogs = load_slogs(asked);
  const Dictionary& dict = load_dictionary_or_throw();
  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const std::unique_ptr<Agent> agent = PlayerFactory::make_player(opt.player, /*thread_id=*/0);
  std::ofstream file;
  if (!opt.out.empty()) {
    file.open(opt.out);
    if (!file) throw util::CleanException("cannot write {}", opt.out);
  }
  std::ostream& out = opt.out.empty() ? std::cout : file;
  for (size_t i = 0; i < asked.size(); ++i) {
    out << json::serialize(ask(*agent, asked[i], slogs.at(asked[i].slog), dict)) << "\n"
        << std::flush;
    if ((i + 1) % 25 == 0 || i + 1 == asked.size())
      std::cerr << std::format("positions: {} / {}\n", i + 1, asked.size());
  }
}

}  // namespace
}  // namespace scribblez

int main(int argc, char** argv) {
  using namespace scribblez;
  try {
    Options opt;
    SeedProducer::Params seed;
    boost::program_options::options_description desc("position_pick_tool options");
    add_options(desc, opt);
    seed.add_options(desc);
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);
    SeedProducer::instance().seed(seed);
    util::set_worker_count(opt.threads);
    run(opt);
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
