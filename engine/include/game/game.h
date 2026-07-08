#pragma once

#include "agent/agent.h"
#include "game/bag.h"
#include "game/board.h"
#include "game/move.h"
#include "game/rack.h"
#include "lexicon/dictionary.h"

#include <array>
#include <string>
#include <vector>

namespace scribblez {

struct TurnRecord {
  int player;  // 0 or 1
  Rack rack_before;
  int bag_size_before;
  Move move;
  int score_delta;  // points scored on this turn (may be 0 or negative)
  std::array<int, 2> cumulative_scores;
  // Tiles drawn from the bag after the move (in draw order). Trailing entries
  // of `drawn` are empty Tiles. A turn draws at most RACK_SIZE tiles.
  Rack drawn;
};

// Non-owning view of one completed game's log. The variable-length backing
// store (the turn array and the name/end-reason strings) lives elsewhere -- a
// GameLogStorage produced by self-play, or a decoder's scratch buffer -- and
// must outlive the view. The fixed-size fields (racks, scores) are held by
// value. This is the single currency the tensorization path consumes, so both
// the self-play and on-disk replay paths build a GameLog and funnel through the
// same encoder.
struct GameLog {
  uint64_t seed = 0;
  std::array<const char*, 2> player_names = {nullptr, nullptr};
  std::array<int, 2> initial_scores = {0, 0};  // head-start handicap, if any
  std::array<Rack, 2> initial_racks;           // tiles dealt to each player at game start
  const TurnRecord* records = nullptr;         // backing store owned elsewhere
  int num_records = 0;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;   // tiles left on each rack at game end
  const char* end_reason = nullptr;  // "out", "stalemate", or "max_turns"
  // Leading plies played uniformly at random via Game::set_random_opening
  // rather than by the seated agents (0 for a normal game). Positions before
  // the last of these have a random move after them and are excluded from
  // training (see binlog::eligible_span).
  int num_random_opening_plies = 0;
};

// Owning backing store for a game's log. Produced by self-play (Game), the
// manual GCG tool, and tests -- anything that must keep a game's data alive
// beyond the producing scope. `view()` yields a non-owning GameLog pointing
// into this storage (valid for as long as the storage object lives and its
// `turns` vector is not reallocated).
struct GameLogStorage {
  uint64_t seed = 0;
  std::array<std::string, 2> player_names;
  std::array<int, 2> initial_scores = {0, 0};
  std::array<Rack, 2> initial_racks;
  std::vector<TurnRecord> turns;
  std::array<int, 2> final_scores = {0, 0};
  std::array<Rack, 2> final_racks;
  std::string end_reason;
  int num_random_opening_plies = 0;  // see GameLog::num_random_opening_plies

  GameLog view() const;
};

class Game {
 public:
  // The Game does not own the agents; the caller (typically play_game) keeps
  // them alive across multiple Game instances. This lets the same human or
  // bot persist (including any per-process state like a WebSession or a
  // background Macondo subprocess) over a series of games.
  Game(Agent& p0, Agent& p1, const Dictionary& dict, uint64_t seed);

  // Give the players a head-start handicap before play begins. Must be called
  // before play(); asserts that no move has been made yet.
  void set_initial_scores(std::array<int, 2> initial_scores);

  // Play the first `plies` turns of the game uniformly at random (via
  // pick_uniform_random_play, seeded from the game seed) instead of asking the
  // seated agents, to reach off-policy positions -- especially unusual rack
  // leaves -- that agent self-play would never visit. Agents still observe the
  // random moves through observe_move(). Must be called before play(); asserts
  // that no move has been made yet.
  void set_random_opening(int plies);

  void play();

  // Play out from a mid-game position (e.g. a Monte-Carlo rollout of an endgame).
  // `board`/`scores` are the post-move state; `known_racks[p]` holds each player's
  // already-known tiles (a leave, or an empty rack to deal fresh); both racks are
  // then refilled to RACK_SIZE from `pool` (the unseen tiles, drawn in the Game's
  // seeded order), and `to_move` plays first. `returned_to_bag` holds tiles that
  // re-enter the bag only after both refills -- the tiles a just-made exchange
  // surrendered, which are available to future draws but not to either player's
  // initial refill. The turn loop, draws, and endgame scoring are identical to
  // play().
  void play_from(const Board& board, std::array<int, 2> scores,
                 const std::array<Rack, 2>& known_racks, const Bag& pool, int to_move,
                 const Rack& returned_to_bag = Rack{});

  // A non-owning view of this game's log. Valid for as long as the Game (and
  // its internal storage) lives and extract_log() has not been called.
  GameLog log() const { return log_.view(); }

  // Move the owning log storage out of the Game (e.g. to hand to a sink that
  // retains it). After this call, log() must not be used.
  GameLogStorage extract_log() { return std::move(log_); }

  // Live game accessors (valid after construction; reflect final state after
  // play() returns). Used by the web front-end to render the end-of-game board.
  const Board& board() const { return board_; }
  int score(int player) const { return scores_[player]; }
  const Rack& rack(int player) const { return racks_[player]; }
  int bag_size() const { return bag_.size(); }

 private:
  Agent* players_[2];
  const Dictionary& dict_;
  uint64_t seed_;
  Bag bag_;
  Board board_;
  Rack racks_[2];
  std::array<int, 2> scores_{0, 0};
  GameLogStorage log_;
  int random_opening_plies_ = 0;
  std::mt19937_64 opening_rng_;  // drives the random-opening move choices

  // Draw from the bag until the player's rack is at RACK_SIZE. If
  // `drawn_out` is non-null, the drawn tiles are added to it.
  void refill_rack(int p, Rack* drawn_out);

  // The move for the current turn: a uniformly-random one while the turn index
  // is within the random opening (tallying it in the log), the seated agent's
  // choice otherwise.
  Move choose_move(int player, const MoveRequest& req);

  // The turn loop shared by play() and play_from(): `start_player` moves first,
  // play continues until the standard end (out / stalemate / max-turns) with the
  // usual endgame scoring, and the final state is written to the log.
  void play_loop(int start_player);
};

}  // namespace scribblez
