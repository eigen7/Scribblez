#include "agent/human_web_agent.h"

#include "agent/agent_options.h"
#include "game/game.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "serve/client_message.h"
#include "serve/web_server.h"
#include "util/exception.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// Shared by from_spec() and options_help(), so the parsed and documented
// options cannot drift.
po::options_description human_options(HumanWebAgent::Params& params) {
  po::options_description desc;
  desc.add_options()                                                                   //
    ("port", po::value<int>(&params.port)->default_value(params.port),                 //
     "engine WebSocket port")                                                          //
    ("vite-port", po::value<int>(&params.vite_port)->default_value(params.vite_port),  //
     "browser UI (Vite) port")                                                         //
    ("web-dir", po::value<std::string>(&params.web_dir)->default_value(params.web_dir),
     "front-end package dir (cwd of `npm run dev`)");
  return desc;
}

}  // namespace

HumanWebAgent::HumanWebAgent(int thread_id, const Params& params, const std::string& my_name,
                             const std::string& opp_name)
    : Agent(thread_id, my_name), opp_name_(opp_name) {
  // The WebSocket server must be listening before Vite starts, because Vite
  // proxies /ws to it.
  session_ = std::make_unique<WebSession>(params.port);
  vite_ =
    std::make_unique<ViteDevServer>(params.web_dir, params.vite_port, params.port, "", "web", 5173);
  std::cerr << "\n  Starting the web UI (npm run dev in " << params.web_dir << ")...\n";
  vite_->wait_until_ready();
  std::cerr << "\n  Human-vs-AI game ready.\n"
            << "  Open  " << vite_->url() << "  in your browser to play.\n\n";
  std::string cmd = "xdg-open " + vite_->url() + " >/dev/null 2>&1 &";
  int rc = std::system(cmd.c_str());  // best-effort; ignore failure
  (void)rc;
}

HumanWebAgent::~HumanWebAgent() {
  // Let the final WebSocket frame flush before the session is destroyed, or
  // the browser may miss the last message on a clean quit.
  if (session_) session_->linger_after_final_message();
}

MoveDecision HumanWebAgent::make_move(const MoveRequest& req) {
  // The UI's move list shows every legal play and, when the bag allows, every
  // distinct exchange, so the user can see whether HastyBot would rather swap
  // than play. Plays keep their indices, so a {"type":"move","index":...}
  // reply selects a play; exchanges follow the plays and come back as
  // {"type":"exchange","letters":...}.
  const std::vector<Move> plays = generate_legal_plays(req);
  std::vector<Move> display_moves = plays;
  const std::vector<Move> exchanges = generate_legal_exchanges(req);
  display_moves.insert(display_moves.end(), exchanges.begin(), exchanges.end());

  // Each displayed move's HastyBot equity, for the cheat-mode move list. The
  // column stays blank when the equity tables failed to load (see from_spec).
  std::vector<double> equities;
  try {
    equities = HastyEquity::instance().equities(display_moves, req.board, req.bag_size,
                                                req.opp_rack, req.my_rack);
  } catch (const std::exception&) {
    equities.clear();
  }

  StateView view(req, name_, opp_name_, display_moves, equities.empty() ? nullptr : &equities);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_->connected() && !session_->wait_for_client()) {
      return Move::pass();
    }
    session_->send_text(msg);
    for (;;) {
      auto in = session_->recv_text();
      if (!in) break;  // disconnected: re-send on reconnect

      boost::json::value parsed;
      try {
        parsed = boost::json::parse(*in);
      } catch (const std::exception&) {
        continue;  // malformed: keep waiting for a usable message
      }
      if (!parsed.is_object()) continue;
      const boost::json::object& obj = parsed.as_object();
      const std::string type = str_field(obj, "type");

      if (type == "move") {
        auto it = obj.find("index");
        if (it != obj.end() && it->value().is_int64()) {
          long idx = it->value().as_int64();
          if (idx >= 0 && size_t(idx) < plays.size()) {
            return plays[size_t(idx)];
          }
        }
      } else if (type == "pass") {
        return Move::pass();
      } else if (type == "exchange") {
        // e.g. {"type":"exchange","letters":"AB?"}; lowercase or '?' is a blank.
        TileCounts tiles;
        for (char c : str_field(obj, "letters")) {
          Tile L = (c == '?' || (c >= 'a' && c <= 'z')) ? BLANK : Tile::from_char(c);
          if (req.my_rack.count(L) > 0) tiles.add(L);
        }
        if (!tiles.empty()) return Move::exchange(tiles);
      }
      // Anything unrecognized or invalid: keep waiting.
    }
  }
}

EndGameResult HumanWebAgent::end_game(const Game& game, int my_seat) {
  // The final board from this seat, with the Play Again / Quit buttons.
  StateView view(game, my_seat, name_, opp_name_, /*your_turn=*/false,
                 /*game_over=*/true);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_->connected() && !session_->wait_for_client()) {
      return {EndGameAction::QUIT};
    }
    session_->send_text(msg);
    for (;;) {
      auto in = session_->recv_text();
      if (!in) break;  // disconnected: re-send on reconnect (or give up above)

      boost::json::value parsed;
      try {
        parsed = boost::json::parse(*in);
      } catch (const std::exception&) {
        continue;
      }
      if (!parsed.is_object()) continue;
      const std::string type = str_field(parsed.as_object(), "type");
      if (type == "play_again") return {EndGameAction::PLAY_AGAIN};
      if (type == "quit") return {EndGameAction::QUIT};
      // Ignore anything else, e.g. a stale move sent before the game ended.
    }
  }
}

std::unique_ptr<HumanWebAgent> HumanWebAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name,
                                                        const std::string& opp_name) {
  Params params;
  po::options_description desc = human_options(params);
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=human options: {}", e.what());
  }

  // Load the default equity tables for the cheat-mode equity column, which no
  // other agent may have loaded (a no-op if one has, or play_game's
  // --leaves-file did). The column is only an annotation, so failing to load
  // them is not fatal.
  try {
    HastyEquity::ensure_initialized(Lexicon::instance().name());
  } catch (const std::exception&) {
  }

  return std::make_unique<HumanWebAgent>(thread_id, params, name, opp_name);
}

std::string HumanWebAgent::options_help() {
  Params params;  // binding targets; only the defaults are read
  return agent_options_help("  A human player driven through the local browser UI.\n",
                            human_options(params));
}

}  // namespace scribblez
