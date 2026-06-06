#include "scribblez/macondo_bot.h"

#include "scribblez/macondo_oracle.h"
#include "scribblez/macondo_oracle_pool.h"
#include "scribblez/move.h"

#include <boost/program_options.hpp>

#include <stdexcept>
#include <string>

namespace scribblez {

HastyBotAgent::HastyBotAgent(int thread_id, const std::string& name)
    : Agent(thread_id, name), oracle_(&MacondoOraclePool::instance().get(thread_id)) {}

Move HastyBotAgent::make_move(const MoveRequest& req) {
  // Delegate to this thread's Macondo subprocess (cached at construction).
  // HastyBot just needs its top pick; the equities for every other play are
  // computed too but only the human player consumes them (for the cheat-mode
  // column).
  auto eval = oracle_->evaluate(req.board, req.my_rack, req.my_score, req.opp_score,
                                req.legal_plays);
  if (eval.best_index >= 0) return req.legal_plays[static_cast<size_t>(eval.best_index)];

  // Macondo's best play wasn't a legal Scribblez play here (e.g. it picked
  // an exchange we don't enumerate). Pass rather than cheat.
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
  return "  Macondo's HastyBot (best static play), shelled out to a persistent\n"
         "  `macondo` process. Path is set process-wide via --macondo.\n"
         "  Options: (none)\n";
}

}  // namespace scribblez
