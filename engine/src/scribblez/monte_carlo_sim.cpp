#include "scribblez/monte_carlo_sim.h"

#include "scribblez/bag.h"
#include "scribblez/game.h"
#include "scribblez/gcg_reader.h"
#include "scribblez/glyph.h"
#include "scribblez/macondo_bot.h"
#include "scribblez/move.h"
#include "scribblez/tile.h"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

namespace scribblez {

namespace {

// The unseen pool from start_player's POV: a full bag minus the tiles on the board
// and in start_player's leave. The opponent's actual rack is unknown, so it stays in
// the pool (the rollout re-samples it -- a clean full draw, since the opponent
// bingoed).
Bag unseen_pool(const Board& board, const Rack& leave, uint64_t seed) {
  Bag pool(seed);
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (!g.is_empty()) pool.remove(g.is_blank() ? BLANK : g.letter());
    }
  for (int i = 0; i < leave.size(); ++i) pool.remove(leave.tiles()[i]);
  return pool;
}

// One rollout (seed g): play to the end and return start_player_final - opp_final.
int rollout(const MonteCarloPosition& pos, const Dictionary& dict, HastyBotAgent& a0,
            HastyBotAgent& a1, uint64_t seed) {
  const int opponent = 1 - pos.start_player;  // bingoed last turn, so plays first
  std::array<Rack, 2> known;
  known[pos.start_player] = pos.leave;  // start_player keeps its leave
  known[opponent] = Rack{};             // the opponent draws a fresh rack from the pool
  const Bag pool = unseen_pool(pos.board, pos.leave, seed);
  Game game(a0, a1, dict, seed);
  game.play_from(pos.board, pos.scores, known, pool, /*to_move=*/opponent);
  const GameLog log = game.log();
  return log.final_scores[pos.start_player] - log.final_scores[opponent];
}

// Worker: plays games {t+1, t+1+threads, ...} (each seeded by its own g, so the
// thread split doesn't affect any game's outcome) and accumulates into *out.
void monte_carlo_worker(const MonteCarloPosition& pos, const Dictionary& dict, int n, int threads,
                        int t, MonteCarloResult* out) {
  HastyBotAgent::Params p0;
  p0.thread_id = t;
  p0.name = "H0";
  HastyBotAgent::Params p1;
  p1.thread_id = t;
  p1.name = "H1";
  HastyBotAgent a0(p0), a1(p1);  // temperature 0 -> deterministic greedy argmax
  for (int g = t + 1; g <= n; g += threads) {
    const int delta = rollout(pos, dict, a0, a1, static_cast<uint64_t>(g));
    ++out->n;
    if (delta > 0)
      ++out->wins;
    else if (delta < 0)
      ++out->losses;
    else
      ++out->draws;
    ++out->delta_hist[delta];
  }
}

}  // namespace

boost::json::object MonteCarloResult::to_json() const {
  boost::json::object hist;
  for (const auto& [delta, count] : delta_hist) hist[std::to_string(delta)] = count;
  boost::json::object o;
  o["start_player"] = start_player;
  o["n"] = n;
  o["wld"] = boost::json::object{{"win", wins}, {"loss", losses}, {"draw", draws}};
  o["score_delta_hist"] = std::move(hist);
  return o;
}

bool parse_monte_carlo_position(const std::string& gcg_text, MonteCarloPosition* out,
                                std::string* error) {
  ParsedGcgGame game;
  if (!read_gcg_text(gcg_text, &game, error)) return false;
  if (game.snapshots.empty() || game.turns.empty()) {
    if (error) *error = "GCG has no turns";
    return false;
  }
  const ParsedGcgSnapshot& final_pos = game.snapshots.back();
  out->board = final_pos.board;
  out->scores = final_pos.scores;
  // turn_player is the seat to act next (the bingoer); start_player is the one that
  // made the final move.
  out->start_player = 1 - final_pos.turn_player;

  const TurnRecord& last = game.turns.back().record;
  if (last.move.type() != MoveType::PLAY) {
    if (error) *error = "final move is not a tile placement";
    return false;
  }
  Rack leave = last.rack_before;
  for (int i = 0; i < last.move.num_glyphs(); ++i) leave.remove(last.move.glyph(i).rack_tile());
  out->leave = leave;
  return true;
}

MonteCarloResult run_monte_carlo(const MonteCarloPosition& pos, const Dictionary& dict, int n,
                                 int threads) {
  threads = std::clamp(threads, 1, std::max(1, n));
  std::vector<MonteCarloResult> partials(threads);
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t)
    workers.emplace_back(monte_carlo_worker, std::cref(pos), std::cref(dict), n, threads, t,
                         &partials[t]);
  for (auto& w : workers) w.join();

  MonteCarloResult total;
  total.start_player = pos.start_player;
  for (const MonteCarloResult& r : partials) {
    total.n += r.n;
    total.wins += r.wins;
    total.losses += r.losses;
    total.draws += r.draws;
    for (const auto& [delta, count] : r.delta_hist) total.delta_hist[delta] += count;
  }
  return total;
}

}  // namespace scribblez
