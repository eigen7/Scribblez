#include "training/position_eval_analysis.h"

#include "data/gcg_reader.h"
#include "encoding/game_state_encoder.h"
#include "game/board.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "game/tile.h"
#include "serve/position_json.h"
#include "util/assert.h"

#include <boost/json.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

// The POV seat and its leave for a parsed post-move position.
struct FinalPosition {
  int start_player = 0;  // seat that made the final move (the evaluated POV)
  Rack leave;            // start_player's leave = final rack_before minus placed tiles
};

// Parse `gcg_text` into `*game` and derive the post-move POV / leave (shared by the
// input encoder and the board bundle, so the two never disagree on the position).
bool parse_final_position(const std::string& gcg_text, ParsedGcgGame* game, FinalPosition* out,
                          std::string* error) {
  if (!read_gcg_text(gcg_text, game, error)) return false;
  if (game->turns.empty() || game->snapshots.empty()) {
    if (error) *error = "GCG has no turns";
    return false;
  }
  const TurnRecord& last = game->turns.back().record;
  if (last.move.type() != MoveType::PLAY) {
    if (error) *error = "final move is not a tile placement";
    return false;
  }
  // turn_player is the seat to act next (the bingoer); the POV is the one that just
  // moved. This matches parse_monte_carlo_position, so the model is evaluated from
  // the same seat the Monte-Carlo ground truth scores.
  out->start_player = 1 - game->snapshots.back().turn_player;
  out->leave = last.rack_before;
  for (int i = 0; i < last.move.num_glyphs(); ++i)
    out->leave.remove(last.move.glyph(i).rack_tile());
  return true;
}

// Replay every recorded move into a fresh encoder, then encode the input from
// `start_player`'s POV holding `leave`. apply_move needs only the moves (not racks),
// so this reproduces the board / scores / last-two-moves state the training replay
// builds; only the rack and the unseen-pool feature depend on `leave`.
bool replay_and_encode(const ParsedGcgGame& game, int start_player, const Rack& leave,
                       const InputEncodingSpec& spec, float* out, std::string* error) {
  (void)error;
  GameStateEncoder enc{spec};
  for (const ParsedGcgTurn& t : game.turns) enc.apply_move(t.record.move);
  RELEASE_ASSERT(enc.active_player() == game.snapshots.back().turn_player);
  if (spec.opp_leave_input) {
    // Open-leaves arm: the opponent's retained leave is reconstructable from
    // their last recorded move; an empty leave (e.g. the penultimate-bingo
    // datasets, whose to-act player kept nothing) is legitimate and encodes
    // as zeros.
    enc.encode_input(start_player, leave, retained_leave(game, 1 - start_player),
                     /*apply_flip=*/false, out);
  } else {
    enc.encode_input(start_player, leave, /*apply_flip=*/false, out);
  }
  return true;
}

// Parse a leave string into a Rack: A-Z (any case) are letters, '?' is a blank,
// spaces are ignored. Fails on any other character or more than RACK_SIZE tiles.
bool parse_leave(const std::string& s, Rack* out, std::string* error) {
  for (char c : s) {
    if (c == ' ') continue;
    if (out->size() >= RACK_SIZE) {
      if (error) *error = std::format("a leave holds at most {} tiles", RACK_SIZE);
      return false;
    }
    if (c == '?') {
      out->add(BLANK);
    } else if (c >= 'A' && c <= 'Z') {
      out->add(Tile::of(c - 'A'));
    } else if (c >= 'a' && c <= 'z') {
      out->add(Tile::of(c - 'a'));
    } else {
      if (error) *error = std::format("invalid tile '{}' (use A-Z, or ? for a blank)", c);
      return false;
    }
  }
  return true;
}

std::string tile_name(int idx) {
  return idx == BLANK.index() ? "?" : std::string(1, char('A' + idx));
}

// Per-tile counts available off the board: the full distribution minus the played
// tiles (a designated blank on the board counts as a blank).
std::array<int, 27> off_board_counts(const Board& board) {
  std::array<int, 27> avail = TILE_COUNTS;
  for (int r = 0; r < BOARD_SIZE; ++r)
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const Glyph g = board.at(r, c);
      if (!g.is_empty()) --avail[g.is_blank() ? BLANK.index() : g.letter().index()];
    }
  return avail;
}

// True iff every tile in `leave` is available off the board (each tile's count does
// not exceed the full distribution minus what is already on the board).
bool leave_available(const Rack& leave, const Board& board, std::string* error) {
  const std::array<int, 27> avail = off_board_counts(board);
  std::array<int, 27> want{};
  for (int i = 0; i < leave.size(); ++i) ++want[leave.tiles()[i].index()];
  for (int t = 0; t < 27; ++t) {
    if (want[t] > avail[t]) {
      if (error) {
        *error =
          std::format("only {} '{}' available off the board", std::max(0, avail[t]), tile_name(t));
      }
      return false;
    }
  }
  return true;
}

}  // namespace

bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error) {
  ParsedGcgGame game;
  FinalPosition pos;
  if (!parse_final_position(gcg_text, &game, &pos, error)) return false;
  return replay_and_encode(game, pos.start_player, pos.leave, spec, out, error);
}

bool encode_position_eval_analysis_input_with_leave(const std::string& gcg_text,
                                                    const std::string& leave_str,
                                                    const InputEncodingSpec& spec, float* out,
                                                    std::string* error) {
  ParsedGcgGame game;
  FinalPosition pos;
  if (!parse_final_position(gcg_text, &game, &pos, error)) return false;

  Rack leave;
  if (!parse_leave(leave_str, &leave, error)) return false;
  if (leave.size() != pos.leave.size()) {
    if (error) {
      *error =
        std::format("leave must have {} tile(s) to match the original leave", pos.leave.size());
    }
    return false;
  }
  if (!leave_available(leave, game.snapshots.back().board, error)) return false;

  return replay_and_encode(game, pos.start_player, leave, spec, out, error);
}

std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error) {
  ParsedGcgGame game;
  FinalPosition pos;
  if (!parse_final_position(gcg_text, &game, &pos, error)) return "";

  const ParsedGcgSnapshot& final_pos = game.snapshots.back();
  const int opp = 1 - pos.start_player;
  const std::string my_name = std::format("Player {}", pos.start_player + 1);
  const std::string opp_name = std::format("Player {}", opp + 1);
  json::object o =
    position_state_object_pov(final_pos.board, pos.leave, final_pos.scores[pos.start_player],
                              final_pos.scores[opp], my_name, opp_name);
  o["start_player"] = pos.start_player;
  o["last_move"] = move_squares(game.turns.back().record.move);
  return json::serialize(o);
}

}  // namespace scribblez
