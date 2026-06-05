#include "scribblez/human_web_agent.h"

#include "scribblez/game.h"
#include "scribblez/macondo.h"
#include "scribblez/tile.h"
#include "scribblez/web_server.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

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

HumanWebAgent::HumanWebAgent(WebSession& session, const std::string& my_name, const std::string& opp_name)
    : session_(session), my_name_(my_name), opp_name_(opp_name) {}

Move HumanWebAgent::make_move(const MoveRequest& req) {
  // Best-effort: ask Macondo to evaluate every legal play so we can show its
  // equity column in the cheat-mode move list. If Macondo isn't configured
  // (e.g. running on a machine without the binary), or the subprocess fails,
  // we just send the position without equities and the front-end leaves the
  // column blank.
  std::vector<std::optional<double>> equities;
  if (Macondo::initialized()) {
    try {
      auto result = Macondo::instance().evaluate(req.board, req.my_rack, req.my_score,
                                                  req.opp_score, req.legal_plays);
      equities = std::move(result.equities);
    } catch (const std::exception&) {
      equities.clear();
    }
  }

  StateView view(req, my_name_, opp_name_, equities.empty() ? nullptr : &equities);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_.connected() && !session_.wait_for_client()) {
      Move m;
      m.type = MoveType::PASS;
      return m;
    }
    session_.send_text(msg);
    for (;;) {
      auto in = session_.recv_text();
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
        Move m;
        m.type = MoveType::PASS;
        return m;
      } else if (type == "exchange") {
        // Optional: front-end may send {"type":"exchange","letters":"AB?"}.
        Move m;
        m.type = MoveType::EXCHANGE;
        int gi = 0;
        for (char c : str_field(obj, "letters")) {
          Tile L = (c == '?' || (c >= 'a' && c <= 'z')) ? BLANK : Tile::from_char(c);
          if (req.my_rack.count(L) > 0 && gi < RACK_SIZE) m.glyphs[gi++] = Glyph::exchanging(L);
        }
        if (gi > 0) return m;
      }
      // Unknown / invalid: keep waiting for a usable message.
    }
  }
}

EndGameResult HumanWebAgent::end_game(const Game& game, int my_seat) {
  // Render the final board from our seat, with no legal-play list (the game
  // is over) and the Play Again / Quit buttons enabled on the front-end.
  StateView view(game, my_seat, my_name_, opp_name_, /*your_turn=*/false,
                 /*game_over=*/true);
  const std::string msg = game_state_json(view);

  for (;;) {
    if (!session_.connected() && !session_.wait_for_client()) {
      // Browser is gone for good: nothing more to do for this human.
      return {EndGameAction::QUIT};
    }
    session_.send_text(msg);
    for (;;) {
      auto in = session_.recv_text();
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
                                                        const std::string& name, WebSession& session,
                                                        const std::string& opp_name) {
  namespace po = boost::program_options;
  po::options_description desc("human options");
  // No agent-specific options at present (kept for symmetry / future use).
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=human options: ") + e.what());
  }
  return std::make_unique<HumanWebAgent>(session, name, opp_name);
}

std::string HumanWebAgent::options_help() {
  return "  A human player driven through the local browser UI.\n"
         "  Options: (none)\n";
}

}  // namespace scribblez
