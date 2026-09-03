#include "training/position_eval_analysis.h"

#include "data/gcg_post_move.h"
#include "data/gcg_reader.h"
#include "encoding/game_state_encoder.h"
#include "game/board.h"
#include "game/rack.h"
#include "game/tile.h"
#include "game/tile_counts.h"
#include "serve/position_json.h"
#include "training/footprint_collapse.h"
#include "util/assert.h"

#include <boost/json.hpp>

#include <cstdint>
#include <format>
#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

// Replay every recorded move into a fresh encoder, then encode the input from
// `start_player`'s POV holding `leave`, with `opp_leave` as the opponent's
// known tiles where the arm takes them. apply_move needs only the moves (not
// racks), so this reproduces the board / scores / last-two-moves state the
// training replay builds; only the rack, opponent-leave, and unseen-pool
// features depend on the leaves.
void replay_and_encode(const ParsedGcgPostMove& pos, const Rack& leave, const Rack& opp_leave,
                       const InputEncodingSpec& spec, float* out) {
  GameStateEncoder enc{spec};
  for (const ParsedGcgTurn& t : pos.game.turns) enc.apply_move(t.record.move);
  RELEASE_ASSERT(enc.active_player() == 1 - pos.start_player);
  if (spec.opp_leave_input) {
    // Open-leaves arm: an empty opponent leave (the penultimate-bingo
    // datasets, whose to-act player kept nothing) is legitimate and encodes
    // as zeros.
    enc.encode_input(pos.start_player, leave, opp_leave, out);
  } else {
    enc.encode_input(pos.start_player, leave, out);
  }
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

std::string tile_name(Tile t) { return t.is_blank() ? "?" : std::string(1, t.to_char()); }

// Parse `leave_str` as `who`'s alternate leave: it must hold as many tiles as
// `original` and every tile must be drawable from `available` (which it is
// removed from, so a second leave validated against the same pool cannot
// double-spend a tile).
bool parse_alternate_leave(const std::string& leave_str, const Rack& original, const char* who,
                           TileCounts* available, Rack* out, std::string* error) {
  if (!parse_leave(leave_str, out, error)) return false;
  if (out->size() != original.size()) {
    if (error) {
      *error = std::format("the {} leave must have {} tile(s) to match the original leave", who,
                           original.size());
    }
    return false;
  }
  for (int i = 0; i < out->size(); ++i) {
    const Tile t = out->tiles()[i];
    if (!available->remove(t)) {
      if (error) {
        *error = std::format("not enough '{}' available off the board for the {} leave",
                             tile_name(t), who);
      }
      return false;
    }
  }
  return true;
}

}  // namespace

bool encode_position_eval_analysis_input(const std::string& gcg_text, const InputEncodingSpec& spec,
                                         float* out, std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return false;
  replay_and_encode(pos, pos.leave, pos.opp_leave, spec, out);
  return true;
}

bool encode_position_eval_analysis_input_with_leaves(const std::string& gcg_text,
                                                     const std::string& leave_str,
                                                     const std::string* opp_leave_str,
                                                     const InputEncodingSpec& spec, float* out,
                                                     std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return false;

  // What the alternates may be drawn from: everything off the board, less a
  // recorded opponent leave that stays in force.
  TileCounts available =
    unseen_counts(pos.board, opp_leave_str == nullptr ? pos.opp_leave : Rack{});
  Rack leave;
  if (!parse_alternate_leave(leave_str, pos.leave, "POV", &available, &leave, error)) return false;
  Rack opp_leave = pos.opp_leave;
  if (opp_leave_str != nullptr) {
    opp_leave = Rack{};
    if (!parse_alternate_leave(*opp_leave_str, pos.opp_leave, "opponent", &available, &opp_leave,
                               error)) {
      return false;
    }
  }
  replay_and_encode(pos, leave, opp_leave, spec, out);
  return true;
}

bool collapse_position_eval_analysis_placement(const std::string& gcg_text,
                                               const InputEncodingSpec& spec, const float* raw,
                                               float* out, std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return false;
  // The opponent (who moves next on pos.board) draws from or holds the unseen
  // pool: 100 tiles less the board and the mover's rack (pos.leave). Feeding it
  // as the availability counts is what makes the collapse's opp marginal match
  // the availability-masked belief the model was trained on -- e.g. a Y-hook with
  // no Y unseen collapses to a hard zero here.
  uint8_t available_counts[27];
  compute_unseen_pool(available_counts, pos.board, pos.leave);
  collapse_footprint_planes(pos.board, *spec.dict, available_counts, raw, out);
  return true;
}

bool masked_position_eval_analysis_placement(const std::string& gcg_text,
                                             const InputEncodingSpec& spec, const float* raw,
                                             float* out, std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return false;
  uint8_t available_counts[27];
  compute_unseen_pool(available_counts, pos.board, pos.leave);
  masked_placement_distributions(pos.board, *spec.dict, available_counts, raw, out);
  return true;
}

bool legal_position_eval_analysis_placement(const std::string& gcg_text,
                                            const InputEncodingSpec& spec, float* out,
                                            std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return false;
  uint8_t available_counts[27];
  compute_unseen_pool(available_counts, pos.board, pos.leave);
  collapse_footprint_legal_cells(pos.board, *spec.dict, available_counts, out);
  return true;
}

std::string position_eval_analysis_board_json(const std::string& gcg_text, std::string* error) {
  ParsedGcgPostMove pos;
  if (!read_gcg_post_move(gcg_text, &pos, error)) return "";

  const int opp = 1 - pos.start_player;
  const std::string my_name = std::format("Player {}", pos.start_player + 1);
  const std::string opp_name = std::format("Player {}", opp + 1);
  json::object o = position_state_object_pov(pos.board, pos.leave, pos.scores[pos.start_player],
                                             pos.scores[opp], my_name, opp_name);
  o["start_player"] = pos.start_player;
  o["last_move"] = move_squares(pos.game.turns.back().record.move);
  o["opp_leave"] = pos.opp_leave.to_string();
  return json::serialize(o);
}

}  // namespace scribblez
