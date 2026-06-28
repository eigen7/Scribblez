#include "scribblez/macondo_bot.h"

#include "scribblez/agent_options.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/lexicon.h"
#include "scribblez/move.h"
#include "scribblez/movegen.h"
#include "scribblez/seed_producer.h"
#include "scribblez/word_map.h"
#include "util/math.h"

#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scribblez {

HastyBotAgent::HastyBotAgent(const Params& params)
    : Agent(params.thread_id, params.name),
      top_k_(params.top_k),
      temperature_(params.temperature),
      temperature_min_bag_(params.temperature_min_bag),
      rng_(params.seed) {
  if (top_k_ < 1) throw std::runtime_error("hastybot: --top-k must be >= 1");
  if (temperature_ < 0.0) throw std::runtime_error("hastybot: --temperature must be >= 0");
  if (temperature_min_bag_ < 0)
    throw std::runtime_error("hastybot: --temperature-min-bag must be >= 0");
}

namespace {

// Canonical total order on distinct PLAY moves (orientation, anchor, placed
// squares, then placed glyphs) -- used only to break exact-equity ties.
bool move_order_less(const Move& a, const Move& b) {
  if (a.horizontal() != b.horizontal()) return a.horizontal() < b.horizontal();
  if (a.start() != b.start()) return a.start() < b.start();
  if (a.square_mask() != b.square_mask()) return a.square_mask() < b.square_mask();
  const int na = a.num_glyphs(), nb = b.num_glyphs();
  if (na != nb) return na < nb;
  for (int i = 0; i < na; ++i) {
    if (a.glyph(i).code() != b.glyph(i).code()) return a.glyph(i).code() < b.glyph(i).code();
  }
  return false;
}

}  // namespace

bool hasty_move_better(double eq_a, const Move& a, double eq_b, const Move& b) {
  if (eq_a != eq_b) return eq_a > eq_b;
  return move_order_less(a, b);
}

Move hasty_best_move_reference(const MoveRequest& req) {
  const std::vector<Move> plays = generate_legal_plays(req);
  const HastyEquity& eq = HastyEquity::instance();
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  bool have = false;
  Move best;
  double best_eq = 0.0;
  for (const Move& m : plays) {
    const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
    if (!have || hasty_move_better(e, m, best_eq, best)) {
      best = m;
      best_eq = e;
      have = true;
    }
  }
  return have ? best : Move::pass();
}

namespace {

// The per-move inputs the equity bounds depend on.
struct BoundInputs {
  int rack_size;
  bool endgame;
  double endgame_term;
  int bag_size;
  std::array<double, RACK_SIZE + 1> leave_by_size;
};

BoundInputs make_bound_inputs(const MoveRequest& req, const HastyEquity& eq) {
  BoundInputs in;
  in.rack_size = req.my_rack.size();
  in.endgame = req.bag_size <= 0;
  in.endgame_term = in.endgame ? 2.0 * static_cast<double>(req.opp_rack.point_value()) : 0.0;
  in.bag_size = req.bag_size;
  eq.best_leaves_by_size(req.my_rack, in.leave_by_size);
  return in;
}

// Admissible equity bound for placing `placed` tiles whose raw score is at most
// `score_bound`: equity = score + leave(rack - placed) + opening(<=0) + peg, so
// adding the best leave of the complementary size (or the endgame term) bounds it.
double equity_bound(const HastyEquity& eq, const BoundInputs& in, int placed, int score_bound) {
  const double leave_term =
    in.endgame ? in.endgame_term
               : in.leave_by_size[in.rack_size - placed] + eq.peg_for_tiles(placed, in.bag_size);
  return static_cast<double>(score_bound) + leave_term;
}

// Best move found so far, with hasty tie-break.
struct BestMove {
  bool have = false;
  Move move;
  double eq = 0.0;
};

// Fold each candidate play's equity into the running best.
void consider_moves(const HastyEquity& eq, const MoveRequest& req, TurnLeaves& leaves,
                    const std::vector<Move>& moves, BestMove& bm) {
  for (const Move& m : moves) {
    const double e = eq.equity(m, req.board, req.bag_size, req.opp_rack, leaves);
    if (!bm.have || hasty_move_better(e, m, bm.eq, bm.move)) {
      bm.move = m;
      bm.eq = e;
      bm.have = true;
    }
  }
}

// Indices into `units` paired with their equity bound, sorted best-first so the
// search can stop as soon as a bound can no longer beat the best move found.
std::vector<std::pair<double, int>> rank_by_bound(const std::vector<double>& bounds) {
  std::vector<std::pair<double, int>> order;
  order.reserve(bounds.size());
  for (int i = 0; i < static_cast<int>(bounds.size()); ++i) order.emplace_back(bounds[i], i);
  std::sort(order.begin(), order.end(),
            [](const auto& x, const auto& y) { return x.first > y.first; });
  return order;
}

// The GADDAG HastyBot path: bound each anchor by its best tile-count, then
// generate anchors best-first with the GADDAG.
Move hasty_best_move_gaddag(const MoveRequest& req) {
  const HastyEquity& eq = HastyEquity::instance();
  ShadowMoveGen smg(req.board, req.dict);
  const std::vector<ShadowAnchor> anchors = smg.anchors(req.my_rack);
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);
  const BoundInputs in = make_bound_inputs(req, eq);

  std::vector<double> bounds(anchors.size());
  for (size_t i = 0; i < anchors.size(); ++i) {
    double best = -1e18;
    for (int e = 1; e <= in.rack_size && e <= kMaxPlayTiles; ++e) {
      if (anchors[i].score_bound_by_size[e] >= 0) {
        best = std::max(best, equity_bound(eq, in, e, anchors[i].score_bound_by_size[e]));
      }
    }
    bounds[i] = best;
  }

  BestMove bm;
  std::vector<Move> moves;
  for (const auto& [bound, i] : rank_by_bound(bounds)) {
    if (bm.have && bound < bm.eq) break;  // no remaining anchor can beat the best move
    moves.clear();
    smg.generate_anchor(anchors[i], req.my_rack, moves);
    consider_moves(eq, req, leaves, moves, bm);
  }
  return bm.have ? bm.move : Move::pass();
}

// Enumerates every non-empty sub-multiset of a rack and, in the same depth-first
// pass, prices each one's leave (the rack minus the subrack) by walking the leave
// KWG one tile at a time. Each subrack's leave value is read off the cursor with
// no hashing -- the whole turn costs a few hundred arc-follows instead of one
// leave hash per subrack. `subracks[k]` and `leaves[k]` are filled in lockstep.
struct SubrackLeaveDFS {
  const LeaveValues& lv;
  const std::array<std::pair<int, int>, 26>& letters;  // (letter index, count), ascending
  int num_letters;
  WmpSubracks& subracks;
  std::array<std::vector<float>, kMaxPlayTiles + 1>& leaves;

  // The cursor over the leave built so far (in ascending letter order, matching
  // the KWG's tile order): `list` is the sibling list to match the next tile in,
  // `index` is the accumulated word index, `val` is the leave value if it ended
  // here, and `alive` is false once a tile went missing. `subrack`/`size` are the
  // complementary placed tiles.
  void recurse(int idx, uint32_t list, uint32_t index, float val, bool alive, BitRack subrack,
               int size) const {
    if (idx == num_letters) {
      if (size >= 1) {
        subracks[size].push_back(subrack);
        leaves[size].push_back(alive ? val : 0.0f);
      }
      return;
    }
    const int letter = letters[idx].first;
    const int cnt = letters[idx].second;
    const uint8_t code = LeaveValues::klv_code(Tile::of(letter));
    // Cursor state after putting 0..cnt copies of this letter into the leave.
    std::array<uint32_t, RACK_SIZE + 1> sl{}, ix{};
    std::array<float, RACK_SIZE + 1> vl{};
    std::array<bool, RACK_SIZE + 1> al{};
    sl[0] = list;
    ix[0] = index;
    vl[0] = val;
    al[0] = alive;
    for (int k = 1; k <= cnt; ++k) {
      if (!al[k - 1]) {
        al[k] = false;
        sl[k] = 0;
        ix[k] = ix[k - 1];
        vl[k] = 0.0f;
        continue;
      }
      uint32_t index_at = ix[k - 1];
      const uint32_t a = lv.klv_step(sl[k - 1], code, &index_at);
      if (a == 0) {
        al[k] = false;
        sl[k] = 0;
        ix[k] = index_at;
        vl[k] = 0.0f;
      } else {
        const bool acc = lv.klv_accepts(a);
        al[k] = true;
        vl[k] = acc ? lv.klv_value_at(index_at) : 0.0f;
        ix[k] = index_at + (acc ? 1u : 0u);
        sl[k] = lv.klv_next(a);
      }
    }
    BitRack sr = subrack;
    for (int placed = 0; placed <= cnt; ++placed) {
      const int leave_count = cnt - placed;  // copies kept -> how far the leave cursor advanced
      recurse(idx + 1, sl[leave_count], ix[leave_count], vl[leave_count], al[leave_count], sr,
              size + placed);
      sr.add_letter(letter);
    }
  }
};

// The WordMap HastyBot path: bound each individual extent, then generate extents
// best-first with WordMap lookups. The finer (per-extent) granularity lets the
// early-exit prune whole extents, the regime where WordMap beats the GADDAG.
//
// Racks holding a blank fall back to the GADDAG path: the WordMap is blank-free,
// and resolving blanks by enumerating their letters at query time is too slow
// (MAGPIE uses precomputed blank tables instead). Blanks are a minority of racks,
// so self-play still gets the WordMap speedup on the common (blank-free) case.
Move hasty_best_move_wmp_impl(const MoveRequest& req, const WordMap& wm) {
  if (req.my_rack.counts().blanks() > 0) return hasty_best_move_gaddag(req);
  const HastyEquity& eq = HastyEquity::instance();
  TurnLeaves leaves = eq.turn_leaves(req.my_rack);

  // Enumerate the rack's subracks and price every subrack's leave in one DFS over
  // the leave KWG (no per-subrack hashing). subracks[k]/leaves[k] are aligned.
  WmpSubracks subracks;
  std::array<std::vector<float>, kMaxPlayTiles + 1> sub_leaves;
  std::array<std::pair<int, int>, 26> letters{};
  int num_letters = 0, rack_tiles = 0;
  const TileCounts& rack_counts = req.my_rack.counts();
  for (Tile L = Tile::of(0); L < 26; ++L) {
    const int c = rack_counts.count(L);
    if (c > 0) {
      letters[num_letters++] = {L.index(), c};
      rack_tiles += c;
    }
  }
  const LeaveValues& lv = eq.leave_table();
  const SubrackLeaveDFS dfs{lv, letters, num_letters, subracks, sub_leaves};
  dfs.recurse(0, lv.klv_root(), /*index=*/0, /*val=*/0.0f, /*alive=*/true, BitRack{}, 0);

  // The shadow bounds need the best leave of each size, which is exactly the max
  // priced leave per subrack size -- so derive it from the DFS instead of a second
  // best-leave enumeration. A play placing `placed` tiles keeps size rack-placed.
  BoundInputs in;
  in.rack_size = rack_tiles;
  in.endgame = req.bag_size <= 0;
  in.endgame_term = in.endgame ? 2.0 * static_cast<double>(req.opp_rack.point_value()) : 0.0;
  in.bag_size = req.bag_size;
  in.leave_by_size.fill(-1e18);
  for (int placed = 1; placed <= rack_tiles && placed <= kMaxPlayTiles; ++placed) {
    double best = -1e18;
    for (float v : sub_leaves[placed]) best = std::max(best, static_cast<double>(v));
    in.leave_by_size[rack_tiles - placed] = best;
  }

  // Derive, per subrack size:
  //   - sub_terms[size][j]: the subrack's equity contribution (leave value +
  //     pre-endgame term), for the per-subrack generation cutoff.
  //   - has_word[size]: whether any size-k subrack forms a k-letter word -- the
  //     shadow's nonplaythrough existence prune (handed to extents()).
  //   - np_best_leave_term[size]: the best leave term among WORD-FORMING subracks
  //     of that size (MAGPIE's nonplaythrough_best_leave_values), a tighter leave
  //     bound for nonplaythrough extents than the global best-leave-by-size.
  // The word-existence lookups here are shadow-time checks, not generation probes.
  std::array<std::vector<double>, kMaxPlayTiles + 1> sub_terms;
  std::array<bool, kMaxPlayTiles + 1> has_word{};
  std::array<double, kMaxPlayTiles + 1> np_best_leave_term;
  np_best_leave_term.fill(-1e18);
  for (int size = 1; size <= rack_tiles && size <= kMaxPlayTiles; ++size) {
    const std::vector<BitRack>& subs = subracks[size];
    const double peg = in.endgame ? 0.0 : eq.peg_for_tiles(size, in.bag_size);
    sub_terms[size].resize(subs.size());
    for (size_t j = 0; j < subs.size(); ++j) {
      const double term = in.endgame ? in.endgame_term : (sub_leaves[size][j] + peg);
      sub_terms[size][j] = term;
      if (size >= 2 && wm.lookup(size, subs[j]).count > 0) {
        has_word[size] = true;
        if (term > np_best_leave_term[size]) np_best_leave_term[size] = term;
      }
    }
  }

  ShadowMoveGen smg(req.board, req.dict);
  const std::vector<ShadowExtent> extents = smg.extents(req.my_rack, &wm, &has_word);

  std::vector<double> bounds(extents.size());
  for (size_t i = 0; i < extents.size(); ++i) {
    const ShadowExtent& e = extents[i];
    // A nonplaythrough extent (word == its placed tiles) is bounded by the best
    // leave among the size-placed subracks that actually form a word; a playthrough
    // extent keeps the looser global best-leave-by-size.
    double leave_term;
    if (e.length == e.placed && has_word[e.placed]) {
      leave_term = np_best_leave_term[e.placed];
    } else if (in.endgame) {
      leave_term = in.endgame_term;
    } else {
      leave_term =
        in.leave_by_size[in.rack_size - e.placed] + eq.peg_for_tiles(e.placed, in.bag_size);
    }
    bounds[i] = static_cast<double>(e.score_bound) + leave_term;
  }

  BestMove bm;
  std::vector<Move> moves;
  for (const auto& [bound, i] : rank_by_bound(bounds)) {
    if (bm.have && bound < bm.eq) break;  // no remaining extent can beat the best move
    const ShadowExtent& e = extents[i];
    moves.clear();
    wmp_generate_extent(req.board, wm, subracks, e, moves, bm.have ? bm.eq : -1e18,
                        sub_terms[e.placed].data());
    consider_moves(eq, req, leaves, moves, bm);
  }
  return bm.have ? bm.move : Move::pass();
}

// The HastyBot --player options, binding the softmax-sampler controls to the
// given storage. from_spec() builds this to parse a spec; options_help() builds
// it against scratch defaults to document the same flags, so the parsed options
// and the documented options share one source of truth.
boost::program_options::options_description hastybot_options(int& top_k, double& temperature,
                                                             int& temperature_min_bag,
                                                             uint64_t& seed) {
  namespace po = boost::program_options;
  po::options_description desc("hastybot options");
  desc.add_options()                                                              //
    ("top-k", po::value<int>(&top_k)->default_value(top_k),                       //
     "candidate moves to sample among when temperature > 0")                      //
    ("temperature", po::value<double>(&temperature)->default_value(temperature),  //
     "softmax sampling temperature over equity (0 = greedy argmax)")              //
    ("temperature-min-bag",
     po::value<int>(&temperature_min_bag)->default_value(temperature_min_bag),
     "sample only while bag >= this many tiles (0 = sample all game)")  //
    ("seed", po::value<uint64_t>(&seed), "sampling PRNG seed (default: SeedProducer)");
  return desc;
}

}  // namespace

Move HastyBotAgent::make_move(const MoveRequest& req) {
  // Greedy: the fast pruned shadow-play search. Used when sampling is disabled
  // (temperature 0) or confined to the opening and the bag has dropped below
  // temperature_min_bag_ (so the rest of the game keeps HastyBot's exact
  // strength).
  if (temperature_ <= 0.0 || req.bag_size < temperature_min_bag_) {
    return hasty_best_move_gaddag(req);
  }

  // Exploratory: generate every legal play, rank by equity, keep the top-K, and
  // softmax-sample among them to inject exploration into self-play data
  // generation that pure argmax play lacks.
  const std::vector<Move> plays = generate_legal_plays(req);
  const int n = static_cast<int>(plays.size());
  if (n == 0) return Move::pass();
  const HastyEquity& eq = HastyEquity::instance();
  const std::vector<double> vals =
    eq.equities(plays, req.board, req.bag_size, req.opp_rack, req.my_rack);

  const int k = std::min(top_k_, n);
  std::vector<int> idx(static_cast<size_t>(n));
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return vals[a] > vals[b]; });
  std::vector<double> top_vals(static_cast<size_t>(k));
  for (int j = 0; j < k; ++j) top_vals[j] = vals[idx[j]];
  const int chosen = sampler_.sample(top_vals, k, temperature_, rng_);
  return plays[static_cast<size_t>(idx[chosen])];
}

Move hasty_best_move_wmp(const MoveRequest& req, const WordMap& wm) {
  return hasty_best_move_wmp_impl(req, wm);
}

std::unique_ptr<HastyBotAgent> HastyBotAgent::from_spec(const std::vector<std::string>& tokens,
                                                        int thread_id, const std::string& name) {
  namespace po = boost::program_options;

  // The equity tables (leaves + pre-endgame) are process-wide: a play_game
  // --leaves-file overrides them, but otherwise we lazily load Macondo's
  // defaults for the active lexicon. The only per-agent options control the
  // optional softmax-over-equity sampler used for exploratory self-play.
  int top_k = 10;
  double temperature = 0.0;
  int temperature_min_bag = 0;
  uint64_t seed = 0;
  bool have_seed = false;

  po::options_description desc = hastybot_options(top_k, temperature, temperature_min_bag, seed);

  try {
    po::variables_map vm;
    po::store(po::command_line_parser(tokens).options(desc).run(), vm);
    po::notify(vm);
    have_seed = vm.count("seed") > 0;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("bad --type=hastybot options: ") + e.what());
  }

  HastyEquity::ensure_initialized(Lexicon::instance().name());
  const uint64_t resolved_seed = have_seed ? seed : SeedProducer::instance().next();
  return std::make_unique<HastyBotAgent>(
    HastyBotAgent::Params{.thread_id = thread_id,
                          .name = name,
                          .top_k = top_k,
                          .temperature = temperature,
                          .seed = resolved_seed,
                          .temperature_min_bag = temperature_min_bag});
}

std::string HastyBotAgent::options_help() {
  int top_k = 10;
  double temperature = 0.0;
  int temperature_min_bag = 0;
  uint64_t seed = 0;  // scratch binding targets; never read here
  return agent_options_help(
    "  In-process HastyBot: enumerates all legal plays and ranks them by\n"
    "  static equity (score + leave value + adjustments).\n",
    hastybot_options(top_k, temperature, temperature_min_bag, seed));
}

}  // namespace scribblez
