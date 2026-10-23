#include "sim/hasty_rollout_task.h"

#include "agent/agent.h"
#include "agent/hasty_bot.h"
#include "game/glyph.h"
#include "game/tile.h"

#include <algorithm>
#include <array>
#include <format>
#include <thread>
#include <utility>

namespace scribblez {

using rollout_abi::PackedMove;
using rollout_abi::PositionAbi;
using rollout_abi::RolloutResult;

HastyRolloutPosition make_rollout_position(const Board& board, const Rack& leave, int my_score,
                                           int opp_score, uint64_t seed) {
  return HastyRolloutPosition{.board = board,
                              .leave = leave,
                              .my_score = my_score,
                              .opp_score = opp_score,
                              .pool = board.unseen_tiles(leave),
                              .seed = seed};
}

namespace {

// The pool as the bag's initial contents: its tiles ascending by tile.
void fill_bag(const TileCounts& pool, rollout_abi::Bag& bag) {
  bag.len = 0;
  bag.cursor = 0;
  bag.next_rand = 0;
  for (int k = 0; k < TILE_KINDS; ++k) {
    for (int c = 0; c < pool.count(Tile::of(k)); ++c) bag.tiles[bag.len++] = uint8_t(k);
  }
}

void draw_into(Rack& rack, rollout_abi::Bag& bag, int n, uint64_t seed, uint32_t rollout) {
  for (int i = 0; i < n; ++i) rack.add(Tile::of(bag.draw(seed, rollout)));
}

}  // namespace

PositionAbi to_abi(const HastyRolloutPosition& pos, const Dictionary& dict) {
  pos.board.ensure_movegen_caches(dict);
  PositionAbi abi;
  for (int i = 0; i < rollout_abi::kSquares; ++i) {
    abi.squares[i] = pos.board.at(i / BOARD_SIZE, i % BOARD_SIZE).code();
  }
  for (int t = 0; t < 2; ++t) {
    const auto& cross = pos.board.cross_checks(t == 1);
    for (int i = 0; i < rollout_abi::kSquares; ++i) {
      abi.cross_mask[t][i] = cross[i].mask;
      abi.cross_score[t][i] = int16_t(cross[i].score);
      abi.cross_has_neighbor[t][i] = cross[i].has_neighbor ? 1 : 0;
    }
  }
  for (int k = 0; k < TILE_KINDS; ++k) abi.leave[k] = uint8_t(pos.leave.count(Tile::of(k)));
  rollout_abi::Bag bag{};
  fill_bag(pos.pool, bag);
  for (int i = 0; i < bag.len; ++i) abi.pool[i] = bag.tiles[i];
  abi.pool_size = bag.len;
  abi.num_tiles = pos.board.num_tiles();
  abi.scores[0] = pos.my_score;
  abi.scores[1] = pos.opp_score;
  abi.seed = pos.seed;
  return abi;
}

PackedMove pack_move(const Move& m) {
  PackedMove p;
  p.type = uint8_t(m.type());
  p.horizontal = m.horizontal() ? 1 : 0;
  p.start = uint8_t(m.start());
  p.num_glyphs = uint8_t(m.num_glyphs());
  for (int i = 0; i < m.num_glyphs(); ++i) p.glyphs[i] = m.glyph(i).code();
  p.square_mask = m.square_mask();
  p.score = m.score();
  return p;
}

Move unpack_move(const PackedMove& p) {
  if (p.type == uint8_t(MoveType::PASS)) return Move::pass();
  if (p.type == uint8_t(MoveType::EXCHANGE)) {
    TileCounts tiles;
    for (int i = 0; i < p.num_glyphs; ++i) {
      tiles.add(p.glyphs[i] == Glyph::blank().code() ? BLANK : Tile::of(p.glyphs[i] - 1));
    }
    return Move::exchange(tiles);
  }
  std::array<Glyph, RACK_SIZE> played{};
  for (int i = 0; i < p.num_glyphs; ++i) {
    played[i] = Glyph::played(Tile::of(rollout_abi::glyph_letter(p.glyphs[i])),
                              rollout_abi::glyph_is_blank(p.glyphs[i]));
  }
  return Move::play(p.horizontal != 0, p.start, p.square_mask, p.score, played.data(),
                    p.num_glyphs);
}

std::string describe_move(const Move& m, const Board& board) {
  if (m.type() == MoveType::PASS) return "pass";
  if (m.type() == MoveType::EXCHANGE) {
    std::string s = "exch ";
    for (int i = 0; i < m.num_glyphs(); ++i) s.push_back(m.glyph(i).to_char());
    return s;
  }
  const auto [r, c] = m.word_origin(board);
  return std::format("{}{} {} {} {}", r + 1, char('A' + c), m.horizontal() ? "across" : "down",
                     m.main_word(board), m.score());
}

namespace {

// One rollout's mutable state, mirroring what the device keeps per block.
struct RolloutState {
  Board board;
  std::array<Rack, 2> racks;   // 0 us, 1 the opponent
  std::array<Rack, 2> leaves;  // each player's rack after its last move, before its draw
  std::array<int, 2> scores;
  rollout_abi::Bag bag;
  bool game_over = false;
};

RolloutState start_rollout(const HastyRolloutPosition& pos, uint32_t rollout) {
  RolloutState st{.board = pos.board,
                  .racks = {pos.leave, Rack{}},
                  .scores = {pos.my_score, pos.opp_score},
                  .bag = {}};
  fill_bag(pos.pool, st.bag);
  const int pool_size = st.bag.len;
  draw_into(st.racks[1], st.bag, rollout_abi::opp_rack_draw(pool_size), pos.seed, rollout);
  draw_into(st.racks[0], st.bag, rollout_abi::our_refill_draw(pool_size, pos.leave.size()),
            pos.seed, rollout);
  st.leaves = st.racks;  // a player who never moves keeps its full rack as its leave
  return st;
}

// One ply by `mover`: HastyBot's move, applied with the draw and the play-out
// bonus. Returns the move.
Move play_ply(RolloutState& st, int mover, const Dictionary& dict, uint64_t seed,
              uint32_t rollout) {
  Rack& rack = st.racks[mover];
  const Rack& other = st.racks[1 - mover];
  const int bag_size = st.bag.remaining();
  const Rack hidden;
  if (rack.empty()) return Move::pass();
  const MoveRequest req{
    st.board, dict, rack, bag_size == 0 ? other : hidden, st.scores[mover], st.scores[1 - mover],
    bag_size};
  const Move m = hasty_best_move_wmp(req);
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  st.leaves[mover] = rack;
  if (m.type() == MoveType::PLAY) {
    st.board.apply(m);
    st.scores[mover] += m.score();
    draw_into(rack, st.bag, std::min(m.num_glyphs(), st.bag.remaining()), seed, rollout);
    if (rack.empty()) {
      st.scores[mover] += 2 * other.point_value();
      st.game_over = true;
    }
  } else if (m.type() == MoveType::EXCHANGE) {
    draw_into(rack, st.bag, m.num_glyphs(), seed, rollout);
    for (int i = 0; i < m.num_glyphs(); ++i) st.bag.put_back(uint8_t(m.glyph(i).rack_tile()));
  }
  return m;
}

RolloutResult run_one(const HastyRolloutPosition& pos, const Dictionary& dict, uint32_t rollout,
                      CpuLeaf* leaf) {
  RolloutState st = start_rollout(pos, rollout);
  RolloutResult out;
  for (int ply = 0; ply < 2; ++ply) {
    if (st.game_over) break;
    const int mover = ply == 0 ? 1 : 0;
    out.moves[ply] = pack_move(play_ply(st, mover, dict, pos.seed, rollout));
  }
  out.game_over = st.game_over ? 1 : 0;
  out.diff = int16_t(st.scores[0] - st.scores[1]);
  if (leaf) *leaf = CpuLeaf{.board = st.board, .leaves = st.leaves, .scores = st.scores};
  return out;
}

}  // namespace

RolloutRacks rollout_racks(const HastyRolloutPosition& pos, uint32_t rollout) {
  const RolloutState st = start_rollout(pos, rollout);
  return RolloutRacks{.opp = st.racks[1], .mine = st.racks[0]};
}

std::vector<RolloutResult> run_hasty_rollouts_cpu(const HastyRolloutPosition& pos,
                                                  const Dictionary& dict, int n, int threads,
                                                  std::vector<CpuLeaf>* leaves_out) {
  pos.board.ensure_movegen_caches(dict);
  std::vector<RolloutResult> results(static_cast<size_t>(n));
  const size_t leaves = leaves_out ? leaves_out->size() : 0;
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      for (int i = t; i < n; i += threads) {
        CpuLeaf* leaf = size_t(i) < leaves ? &(*leaves_out)[size_t(i)] : nullptr;
        results[size_t(i)] = run_one(pos, dict, uint32_t(i), leaf);
      }
    });
  }
  for (std::thread& w : workers) w.join();
  return results;
}

}  // namespace scribblez
