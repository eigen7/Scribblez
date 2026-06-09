#include "scribblez/human_web_agent.h"

#include "scribblez/game.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/tile.h"
#include "scribblez/web_server.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace scribblez {

namespace {

// Best-effort string field from a parsed client message (empty if absent or
// not a string).
std::string str_field(const boost::json::object& obj, boost::json::string_view key) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->value().is_string()) return "";
  return std::string(it->value().as_string().c_str());
}

}  // namespace

HumanWebAgent::HumanWebAgent(int thread_id, const Params& params, const std::string& my_name,
                             const std::string& opp_name)
    : Agent(thread_id, my_name), opp_name_(opp_name) {
  // The order matters: the WebSocket server must be bound (so its port is
  // listening) before we launch Vite, since Vite proxies /ws to it. Then we
  // block until Vite is accepting browser connections, and best-effort open
  // the URL in the user's browser.
  session_ = std::make_unique<WebSession>(params.port);
  vite_ = std::make_unique<ViteDevServer>(params.web_dir, params.vite_port, params.port);
  std::cerr << "\n  Starting the web UI (npm run dev in " << params.web_dir << ")...\n";
  if (!vite_->wait_until_ready()) {
    throw std::runtime_error("the Vite dev server did not start. See " + params.web_dir +
                             "/.vite-dev.log for details. Did you run ./build.py to install "
                             "the web dependencies?");
  }
  std::cerr << "\n  Human-vs-AI game ready.\n"
            << "  Open  " << vite_->url() << "  in your browser to play.\n\n";
  std::string cmd = "xdg-open " + vite_->url() + " >/dev/null 2>&1 &";
  int rc = std::system(cmd.c_str());  // best-effort; ignore failure
  (void)rc;
}

HumanWebAgent::~HumanWebAgent() {
  // Give the kernel a moment to flush any final WebSocket frame before the
  // session is destroyed; otherwise the browser may miss the last message
  // (e.g. "Thanks for playing.") on a clean quit.
  if (session_) session_->linger_after_final_message();
}

Move HumanWebAgent::make_move(const MoveRequest& req) {
  // Annotate each legal play with its HastyBot static equity for the cheat-
  // mode move list.  If HastyEquity was not initialised (--leaves-file absent)
  // the column is left blank on the front-end.
  std::vector<std::optional<double>> equities;
  try {
    const HastyEquity& eq = HastyEquity::instance();
    const std::vector<double> vals =
      eq.equities(req.legal_plays, req.board, req.bag_size, req.opp_rack);
    equities.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) equities[i] = vals[i];
  } catch (const std::exception&) {
    equities.clear();
  }

  StateView view(req, name_, opp_name_, equities.empty() ? nullptr : &equities);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_->connected() && !session_->wait_for_client()) {
      return MoveFactory::pass();
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
          long idx = static_cast<long>(it->value().as_int64());
          if (idx >= 0 && static_cast<size_t>(idx) < req.legal_plays.size()) {
            return req.legal_plays[static_cast<size_t>(idx)];
          }
        }
      } else if (type == "pass") {
        return MoveFactory::pass();
      } else if (type == "exchange") {
        // Optional: front-end may send {"type":"exchange","letters":"AB?"}.
        TileCounts tiles;
        for (char c : str_field(obj, "letters")) {
          Tile L = (c == '?' || (c >= 'a' && c <= 'z')) ? BLANK : Tile::from_char(c);
          if (req.my_rack.count(L) > 0) tiles.add(L);
        }
        if (!tiles.empty()) return MoveFactory::exchange(tiles);
      }
      // Unknown / invalid: keep waiting for a usable message.
    }
  }
}

EndGameResult HumanWebAgent::end_game(const Game& game, int my_seat) {
  // Render the final board from our seat, with no legal-play list (the game
  // is over) and the Play Again / Quit buttons enabled on the front-end.
  StateView view(game, my_seat, name_, opp_name_, /*your_turn=*/false,
                 /*game_over=*/true);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_->connected() && !session_->wait_for_client()) {
      // Browser is gone for good: nothing more to do for this human.
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
      // Any other message (stale move/pass/exchange from before the game
      // ended) is ignored; keep waiting for an explicit choice.
    }
  }
}

std::unique_ptr<HumanWebAgent> HumanWebAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name,
                                                        const std::string& opp_name) {
  namespace po = boost::program_options;
  Params params;
  po::options_description desc("human options");
  desc.add_options()                                                                   //
    ("port", po::value<int>(&params.port)->default_value(params.port),                 //
     "engine WebSocket port")                                                          //
    ("vite-port", po::value<int>(&params.vite_port)->default_value(params.vite_port),  //
     "browser UI (Vite) port")                                                         //
    ("web-dir", po::value<std::string>(&params.web_dir)->default_value(params.web_dir),
     "front-end package dir (cwd of `npm run dev`)");
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=human options: ") + e.what());
  }

  // Lazily load the default equity tables so the cheat-mode equity column is
  // populated even when the opponent is not a HastyBot (which would otherwise
  // be the only thing that initializes them). A play_game --leaves-file has
  // already initialized them, in which case this is a no-op. Equity is purely
  // an optional annotation for a human, so a missing default leaves file is
  // non-fatal -- the column simply stays blank.
  try {
    HastyEquity::ensure_initialized(Lexicon::instance().name());
  } catch (const std::exception&) {
  }

  return std::make_unique<HumanWebAgent>(thread_id, params, name, opp_name);
}

std::string HumanWebAgent::options_help() {
  return "  A human player driven through the local browser UI.\n"
         "  Options:\n"
         "    --port=N        engine WebSocket port (default 8080)\n"
         "    --vite-port=N   browser UI (Vite) port (default 5173)\n"
         "    --web-dir=DIR   front-end package dir (default \"web\")\n";
}

}  // namespace scribblez
