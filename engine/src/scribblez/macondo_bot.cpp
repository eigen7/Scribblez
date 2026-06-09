#include "scribblez/macondo_bot.h"

#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/move.h"

#include <boost/program_options.hpp>

#include <stdexcept>
#include <string>

namespace scribblez {

HastyBotAgent::HastyBotAgent(int thread_id, const std::string& name) : Agent(thread_id, name) {}

Move HastyBotAgent::make_move(const MoveRequest& req) {
  const HastyEquity& eq = HastyEquity::instance();
  const std::vector<double> vals =
    eq.equities(req.legal_plays, req.board, req.bag_size, req.opp_rack);
  int best = -1;
  double best_equity = -1e18;
  for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
    double e = vals[i];
    if (e > best_equity) {
      best_equity = e;
      best = i;
    }
  }
  if (best >= 0) return req.legal_plays[static_cast<size_t>(best)];

  return MoveFactory::pass();
}

std::unique_ptr<HastyBotAgent> HastyBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  namespace po = boost::program_options;

  // No agent-specific options at present. The equity tables (leaves +
  // pre-endgame) are process-wide: a play_game --leaves-file overrides them,
  // but otherwise we lazily load Macondo's defaults for the active lexicon, so
  // running a HastyBot never requires extra command-line flags.
  po::options_description desc("hastybot options");
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=hastybot options: ") + e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  return std::make_unique<HastyBotAgent>(thread_id, name);
}

std::string HastyBotAgent::options_help() {
  return "  In-process HastyBot: enumerates all legal plays and picks the one\n"
         "  with highest static equity (score + leave value + adjustments).\n"
         "  Options: (none)\n";
}

}  // namespace scribblez
