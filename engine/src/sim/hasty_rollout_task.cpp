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

void RolloutBatch::add_rollouts(uint32_t start, uint32_t first, uint32_t count) {
  for (uint32_t i = first; i < first + count; ++i) tasks.push_back({start, i});
}

RolloutGroup make_rollout_group(const Board& board, const Rack& rack, uint64_t seed) {
  return RolloutGroup{.pool = board.unseen_tiles(rack), .seed = seed, .opp_leaves = {}};
}

RolloutStart make_rollout_start(const Board& board, const Rack& rack, int my_score, int opp_score,
                                const Move& move, uint32_t group) {
  RolloutStart s{.board = board,
                 .leave = rack,
                 .returned = {},
                 .my_score = my_score + move.score(),
                 .opp_score = opp_score,
                 .group = group};
  for (int i = 0; i < move.num_glyphs(); ++i) s.leave.remove(move.glyph(i).rack_tile());
  if (move.type() == MoveType::PLAY) {
    s.board.apply(move);
  } else if (move.type() == MoveType::EXCHANGE) {
    for (int i = 0; i < move.num_glyphs(); ++i) s.returned.add(move.glyph(i).rack_tile());
  }
  return s;
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

PositionAbi to_abi(const RolloutStart& start, const Dictionary& dict) {
  start.board.ensure_movegen_caches(dict);
  PositionAbi abi;
  for (int i = 0; i < rollout_abi::kSquares; ++i) {
    abi.squares[i] = start.board.at(i / BOARD_SIZE, i % BOARD_SIZE).code();
  }
  for (int t = 0; t < 2; ++t) {
    const auto& cross = start.board.cross_checks(t == 1);
    for (int i = 0; i < rollout_abi::kSquares; ++i) {
      abi.cross_mask[t][i] = cross[i].mask;
      abi.cross_score[t][i] = int16_t(cross[i].score);
      abi.cross_has_neighbor[t][i] = cross[i].has_neighbor ? 1 : 0;
    }
  }
  for (int k = 0; k < TILE_KINDS; ++k) abi.leave[k] = uint8_t(start.leave.count(Tile::of(k)));
  for (int i = 0; i < start.returned.size(); ++i)
    abi.returned[i] = uint8_t(start.returned.tiles()[size_t(i)].index());
  abi.num_returned = uint8_t(start.returned.size());
  abi.num_tiles = start.board.num_tiles();
  abi.scores[0] = start.my_score;
  abi.scores[1] = start.opp_score;
  abi.group = start.group;
  return abi;
}

rollout_abi::GroupAbi to_abi(const RolloutGroup& group, uint32_t leave_base) {
  rollout_abi::GroupAbi abi;
  rollout_abi::Bag bag{};
  fill_bag(group.pool, bag);
  for (int i = 0; i < bag.len; ++i) abi.pool[i] = bag.tiles[i];
  abi.pool_size = bag.len;
  abi.seed = group.seed;
  abi.leave_base = leave_base;
  abi.num_leaves = uint32_t(group.opp_leaves.size());
  return abi;
}

rollout_abi::OppLeaveAbi to_abi(const Rack& leave) {
  rollout_abi::OppLeaveAbi abi;
  for (int i = 0; i < leave.size(); ++i) abi.tiles[i] = uint8_t(leave.tiles()[size_t(i)].index());
  abi.n = uint8_t(leave.size());
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
  std::array<Rack, 2> racks;     // 0 us, 1 the opponent
  std::array<Rack, 2> leaves{};  // each player's rack after its last move, before its draw
  std::array<int, 2> scores;
  rollout_abi::Bag bag;
  int scoreless = 0;  // consecutive passes and exchanges
  bool game_over = false;
};

// The racks and the bag, in rollout_abi.h's order.
RolloutState start_rollout(const RolloutBatch& batch, const rollout_abi::RolloutTask& task) {
  const RolloutStart& start = batch.starts[task.position];
  const RolloutGroup& group = batch.groups[start.group];
  const Rack opp_leave =
    task.rollout < group.opp_leaves.size() ? group.opp_leaves[task.rollout] : Rack{};
  RolloutState st{.board = start.board,
                  .racks = {start.leave, opp_leave},
                  .scores = {start.my_score, start.opp_score},
                  .bag = {}};
  fill_bag(group.pool, st.bag);
  for (int i = 0; i < opp_leave.size(); ++i)
    rollout_abi::remove_from_bag(st.bag, uint8_t(opp_leave.tiles()[size_t(i)].index()));
  for (int p : {1, 0}) {
    draw_into(st.racks[p], st.bag, rollout_abi::refill_draw(st.bag.remaining(), st.racks[p].size()),
              group.seed, task.rollout);
  }
  for (int i = 0; i < start.returned.size(); ++i)
    st.bag.put_back(uint8_t(start.returned.tiles()[size_t(i)].index()));
  st.leaves = st.racks;  // a player who never moves keeps its full rack as its leave
  // Our move went out with the bag empty: the game ended with it.
  if (st.racks[0].empty()) {
    st.scores[0] += 2 * st.racks[1].point_value();
    st.game_over = true;
  }
  return st;
}

// With the bag empty, a second scoreless turn in a row ends the game
// (rollout_abi.h): each player loses its rack's value.
void end_if_stalemate(RolloutState& st) {
  if (st.bag.remaining() > 0 || st.scoreless < 2) return;
  for (int p = 0; p < 2; ++p) st.scores[p] -= st.racks[p].point_value();
  st.game_over = true;
}

// HastyBot's move for `mover`, or a pass from an empty rack.
Move hasty_move(const RolloutState& st, int mover, const Dictionary& dict) {
  const Rack& rack = st.racks[mover];
  if (rack.empty()) return Move::pass();
  const int bag_size = st.bag.remaining();
  const Rack hidden;
  const MoveRequest req{st.board,
                        dict,
                        rack,
                        bag_size == 0 ? st.racks[1 - mover] : hidden,
                        st.scores[mover],
                        st.scores[1 - mover],
                        bag_size};
  return hasty_best_move_wmp(req);
}

// One ply by `mover`: HastyBot's move, applied with the draw, the play-out
// bonus and the stalemate rule. Returns the move.
Move play_ply(RolloutState& st, int mover, const Dictionary& dict, uint64_t seed,
              uint32_t rollout) {
  Rack& rack = st.racks[mover];
  const Rack& other = st.racks[1 - mover];
  const Move m = hasty_move(st, mover, dict);
  for (int i = 0; i < m.num_glyphs(); ++i) rack.remove(m.glyph(i).rack_tile());
  st.leaves[mover] = rack;
  if (m.type() == MoveType::PLAY) {
    st.board.apply(m);
    st.scores[mover] += m.score();
    st.scoreless = 0;
    draw_into(rack, st.bag, std::min(m.num_glyphs(), st.bag.remaining()), seed, rollout);
    if (rack.empty()) {
      st.scores[mover] += 2 * other.point_value();
      st.game_over = true;
    }
  } else if (m.type() == MoveType::EXCHANGE) {
    draw_into(rack, st.bag, m.num_glyphs(), seed, rollout);
    for (int i = 0; i < m.num_glyphs(); ++i) st.bag.put_back(uint8_t(m.glyph(i).rack_tile()));
    ++st.scoreless;
  } else {
    ++st.scoreless;
  }
  end_if_stalemate(st);
  return m;
}

RolloutResult run_one(const RolloutBatch& batch, const rollout_abi::RolloutTask& task,
                      const Dictionary& dict, CpuLeaf* leaf) {
  RolloutState st = start_rollout(batch, task);
  const uint64_t seed = batch.groups[batch.starts[task.position].group].seed;
  RolloutResult out;
  // Two plies, then to the game's end if the bag is empty (rollout_abi.h).
  for (int ply = 0;
       ply < rollout_abi::kMaxRolloutPlies && !st.game_over && (ply < 2 || st.bag.remaining() == 0);
       ++ply) {
    const int mover = ply % 2 == 0 ? 1 : 0;
    const Move m = play_ply(st, mover, dict, seed, task.rollout);
    if (ply < 2) out.moves[ply] = pack_move(m);
  }
  out.game_over = st.game_over ? 1 : 0;
  out.diff = int16_t(st.scores[0] - st.scores[1]);
  if (leaf) *leaf = CpuLeaf{.board = st.board, .leaves = st.leaves, .scores = st.scores};
  return out;
}

}  // namespace

RolloutRacks rollout_racks(const RolloutBatch& batch, const rollout_abi::RolloutTask& task) {
  const RolloutState st = start_rollout(batch, task);
  return RolloutRacks{.opp = st.racks[1], .mine = st.racks[0]};
}

void for_each_hasty_rollout_cpu(const RolloutBatch& batch, const Dictionary& dict, int threads,
                                const RolloutVisitor& each) {
  for (const RolloutStart& s : batch.starts) s.board.ensure_movegen_caches(dict);
  const size_t n = batch.tasks.size();
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      CpuLeaf leaf;
      for (size_t i = size_t(t); i < n; i += size_t(threads)) {
        const RolloutResult result = run_one(batch, batch.tasks[i], dict, &leaf);
        each(i, result, leaf);
      }
    });
  }
  for (std::thread& w : workers) w.join();
}

std::vector<RolloutResult> run_hasty_rollouts_cpu(const RolloutBatch& batch, const Dictionary& dict,
                                                  int threads, std::vector<CpuLeaf>* leaves_out) {
  std::vector<RolloutResult> results(batch.tasks.size());
  const size_t leaves = leaves_out ? leaves_out->size() : 0;
  for_each_hasty_rollout_cpu(batch, dict, threads,
                             [&](size_t k, const RolloutResult& result, const CpuLeaf& leaf) {
                               results[k] = result;
                               if (k < leaves) (*leaves_out)[k] = leaf;
                             });
  return results;
}

}  // namespace scribblez
