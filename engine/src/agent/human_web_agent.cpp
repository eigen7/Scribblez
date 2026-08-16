#include "agent/human_web_agent.h"

#include "agent/agent_options.h"
#include "game/game.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "serve/web_server.h"
#include "util/exception.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

namespace {

namespace po = boost::program_options;

// The command-line options for `--type=human`, binding each flag to a field of
// `params`. Both from_spec() (to parse) and options_help() (to document) build
// this, so the two share one source of truth for the option list.
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

// Best-effort string field from a parsed client message (empty if absent or
// not a string).
std::string str_field(const boost::json::object& obj, boost::json::string_view key) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->value().is_string()) return "";
  return std::string(it->value().as_string().c_str());
}

// Append an EXCHANGE move for every distinct non-empty sub-multiset of
// `available`, recursing one tile type at a time. `chosen` accumulates the
// tiles surrendered so far and is restored before returning.
void enumerate_exchanges(std::vector<Move>& out, const std::vector<std::pair<Tile, int>>& types,
                         size_t idx, TileCounts& chosen) {
  if (idx == types.size()) {
    if (!chosen.empty()) out.push_back(Move::exchange(chosen));
    return;
  }
  const auto [tile, max_count] = types[idx];
  for (int k = 0; k <= max_count; ++k) {
    enumerate_exchanges(out, types, idx + 1, chosen);
    if (k < max_count) chosen.add(tile);
  }
  for (int k = 0; k < max_count; ++k) chosen.remove(tile);
}

// Append one EXCHANGE move per distinct non-empty subset of `rack` to `out`,
// so the UI's move list can show exchanges (with their HastyBot equity)
// alongside plays. The caller gates this on the bag being large enough to
// exchange at all.
void append_exchange_moves(std::vector<Move>& out, const Rack& rack) {
  const TileCounts counts = rack.counts();
  std::vector<std::pair<Tile, int>> types;
  for (Tile t = Tile::of(0); t <= BLANK; ++t) {
    const int c = counts.count(t);
    if (c > 0) types.emplace_back(t, c);
  }
  TileCounts chosen;
  enumerate_exchanges(out, types, 0, chosen);
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
  vite_ =
    std::make_unique<ViteDevServer>(params.web_dir, params.vite_port, params.port, "", "web", 5173);
  std::cerr << "\n  Starting the web UI (npm run dev in " << params.web_dir << ")...\n";
  if (!vite_->wait_until_ready()) {
    throw util::CleanException(
      "the Vite dev server did not start. See {}/.vite-dev.log for details. Did you run "
      "py/build.py to install the web dependencies?",
      params.web_dir);
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

MoveDecision HumanWebAgent::make_move(const MoveRequest& req) {
  // The UI's move list shows every legal play plus -- when exchanging is even
  // possible (bag has a full rack to draw from) -- every distinct exchange, so
  // the human can see whether HastyBot would rather swap tiles than play a
  // word. Plays keep their original indices so a `{"type":"move","index":...}`
  // reply still selects the right play; exchanges sit past the end of the play
  // list and are submitted via the `{"type":"exchange","letters":...}` path.
  const std::vector<Move> plays = generate_legal_plays(req);
  std::vector<Move> display_moves = plays;
  if (req.bag_size >= RACK_SIZE) append_exchange_moves(display_moves, req.my_rack);

  // Annotate each displayed move with its HastyBot static equity for the
  // cheat-mode move list.  If HastyEquity was not initialised (--leaves-file
  // absent) the column is left blank on the front-end.
  std::vector<std::optional<double>> equities;
  try {
    const HastyEquity& eq = HastyEquity::instance();
    const std::vector<double> vals =
      eq.equities(display_moves, req.board, req.bag_size, req.opp_rack, req.my_rack);
    equities.resize(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) equities[i] = vals[i];
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
        // Optional: front-end may send {"type":"exchange","letters":"AB?"}.
        TileCounts tiles;
        for (char c : str_field(obj, "letters")) {
          Tile L = (c == '?' || (c >= 'a' && c <= 'z')) ? BLANK : Tile::from_char(c);
          if (req.my_rack.count(L) > 0) tiles.add(L);
        }
        if (!tiles.empty()) return Move::exchange(tiles);
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
  Params params;
  po::options_description desc = human_options(params);
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw util::CleanException("bad --type=human options: {}", e.what());
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
  Params params;  // defaults only; used to render the option list and its defaults
  return agent_options_help("  A human player driven through the local browser UI.\n",
                            human_options(params));
}

}  // namespace scribblez
