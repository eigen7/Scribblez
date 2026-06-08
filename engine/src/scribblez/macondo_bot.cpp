#include "scribblez/macondo_bot.h"

#include "scribblez/hasty_equity.h"
#include "scribblez/move.h"

#include <boost/program_options.hpp>

#include <stdexcept>
#include <string>

namespace scribblez {

HastyBotAgent::HastyBotAgent(int thread_id, const std::string& name) : Agent(thread_id, name) {}

Move HastyBotAgent::make_move(const MoveRequest& req) {
  const HastyEquity& eq = HastyEquity::instance();
  int best = -1;
  double best_equity = -1e18;
  for (int i = 0; i < static_cast<int>(req.legal_plays.size()); ++i) {
    double e = eq.equity(req.legal_plays[i], req.board, req.bag_size, req.my_rack, req.opp_rack);
    if (e > best_equity) {
      best_equity = e;
      best = i;
    }
  }
  if (best >= 0) return req.legal_plays[static_cast<size_t>(best)];

  Move m;
  m.type = MoveType::PASS;
  return m;
}

std::unique_ptr<HastyBotAgent> HastyBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  namespace po = boost::program_options;

  // No agent-specific options at present: the path to the macondo binary is
  // a process-wide option (--macondo) parsed by play_game, since the human
  // player also uses Macondo (for equity annotations) and it'd be silly to
  // re-specify the path per seat.
  po::options_description desc("hastybot options");
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=hastybot options: ") + e.what());
  }

  return std::make_unique<HastyBotAgent>(thread_id, name);
}

std::string HastyBotAgent::options_help() {
  return "  In-process HastyBot: enumerates all legal plays and picks the one\n"
         "  with highest static equity (score + leave value + adjustments).\n"
         "  Options: (none)\n";
}

}  // namespace scribblez
