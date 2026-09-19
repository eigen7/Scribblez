#include "sim/slog_position_simmer.h"

#include "agent/agent.h"
#include "data/binary_log.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "sim/setup_plays.h"
#include "util/exception.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <thread>

namespace scribblez {

namespace {

int32_t equity_rank(const std::vector<Move>& ranked, const Move& m) {
  const auto it = std::find(ranked.begin(), ranked.end(), m);
  return it == ranked.end() ? -1 : int32_t(it - ranked.begin());
}

// The pre-move decision point `encoder` is replayed to, as the sims see it.
SimPosition sim_position(const binlog::PositionEncoder& encoder, const GameLog& g, int turn_idx,
                         int mover, bool open_leaves) {
  SimPosition pos;
  pos.board = encoder.enc().board();
  pos.scores = {encoder.enc().score(0), encoder.enc().score(1)};
  pos.mover = mover;
  pos.rack = encoder.rack(mover);
  // Open leaves: the replay knows both the opponent's rack and the draws that
  // followed their last move, so their retained leave -- the Bayesian-inferable
  // part -- is exact; their replenishments stay hidden and are sampled per
  // rollout.
  if (open_leaves)
    pos.opp_leave = binlog::opp_leave_from_replay(g, turn_idx, encoder.rack(1 - mover));
  return pos;
}

// Every legal move at `pos`, best static equity first. Hidden mode: the
// opponent's replayed rack is ground truth the mover cannot see, so the ranking
// must not use it. Open-leaves mode legitimately reveals the retained leave
// (only equity's endgame adjustments read it).
std::vector<Move> rank_candidates(const SimPosition& pos, const Dictionary& dict, int bag_size) {
  MoveRequest req{
    pos.board, dict, pos.rack, pos.opp_leave, pos.scores[pos.mover], pos.scores[1 - pos.mover],
    bag_size};
  return equity_top_k(req, std::numeric_limits<int>::max());
}

// Shared state of one sim_slog_positions call.
struct SimJob {
  const char* buf;
  const Dictionary& dict;
  const SlogSimConfig& config;
  const std::vector<binlog::GamePositionIndex>& work;
  bool run_sims;
  std::atomic<size_t> next{0};
  std::vector<SimmedPosition> results;
  util::ProgressMeter* meter;
};

void sim_one_position(const binlog::GamePositionIndex& w, SimJob* job,
                      binlog::PositionEncoder* encoder, std::vector<TurnRecord>* scratch,
                      const SimRunner& runner, SimmedPosition* res) {
  const GameLog g = binlog::make_game_view(job->buf, w.game_idx, *scratch, nullptr);
  const int mover = encoder->replay_to_sampled(g, int(w.turn_idx), /*post_move=*/false);
  const SimPosition pos =
    sim_position(*encoder, g, int(w.turn_idx), mover, job->config.open_leaves);
  res->pos = w;
  res->position = pos;
  const uint64_t position_seed = binlog::position_seed(job->config.seed, w.game_idx, w.turn_idx);
  res->base_seed = position_seed + job->config.rollout_seed_offset;
  std::mt19937_64 rng(position_seed);
  res->candidates = job->config.selector(pos, rank_candidates(pos, job->dict, encoder->bag_size()),
                                         g.records[w.turn_idx].move, rng);
  if (job->run_sims && !res->candidates.moves.empty())
    res->observations = runner.run(pos, res->candidates.moves, res->base_seed);
}

// Claims positions off the shared index and fills their result slots. Each
// worker owns its replay scratch and a single-threaded SimRunner.
void run_position_worker(SimJob* job) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(InputEncodingSpec{&job->dict});
  const SimRunner runner(job->dict, job->config.runner);
  const size_t n = job->work.size();
  for (size_t i = job->next.fetch_add(1); i < n; i = job->next.fetch_add(1)) {
    const binlog::GamePositionIndex& w = job->work[i];
    // Name the position a runtime failure (e.g. the leaf-model NaN guard) hit,
    // so an unattended multi-file run leaves a lead instead of a bare message.
    try {
      sim_one_position(w, job, &encoder, &scratch, runner, &job->results[i]);
    } catch (const std::exception& e) {
      throw util::Exception("game {} turn {}: {}", w.game_idx, w.turn_idx, e.what());
    }
    job->meter->add_done();
  }
}

// Thread entry: captures any exception into *err for the joining thread to
// rethrow. A non-finite leaf readout makes SimRunner::run throw at runtime, and
// letting it escape a std::thread would terminate the process instead of
// printing an error.
void position_worker(SimJob* job, std::exception_ptr* err) {
  try {
    run_position_worker(job);
  } catch (...) {
    *err = std::current_exception();
  }
}

}  // namespace

SimCandidates select_sim_candidates(const std::vector<Move>& ranked, const Move& played,
                                    const SimCandidateRecipe& recipe, std::mt19937_64& rng) {
  SimCandidates out;
  out.num_legal_moves = ranked.size();
  if (recipe.quotas) {
    out.moves =
      move_set_eval::stratified_candidates(ranked, played, *recipe.quotas, rng).candidates;
  } else {
    const size_t k = std::min(ranked.size(), size_t(recipe.top_k));
    out.moves.assign(ranked.begin(), ranked.begin() + k);
  }
  for (const Move& m : out.moves) out.equity_ranks.push_back(equity_rank(ranked, m));
  out.highlighted.assign(out.moves.size(), 0);
  return out;
}

SimCandidateSelector recipe_selector(const SimCandidateRecipe& recipe) {
  return
    [recipe](const SimPosition&, const std::vector<Move>& ranked, const Move& played,
             std::mt19937_64& rng) { return select_sim_candidates(ranked, played, recipe, rng); };
}

namespace {

SimCandidates select_setup_candidates(const SimPosition& pos, const Dictionary& dict,
                                      const std::vector<Move>& ranked, int cut, int max_setups) {
  SimCandidates out;
  out.num_legal_moves = ranked.size();
  int setups_outside_cut = 0;
  for (size_t i = 0; i < ranked.size(); ++i) {
    const bool setup = is_high_value_setup(pos.board, dict, pos.rack, ranked[i]);
    const bool outside = int(i) >= cut;
    if (outside && !(setup && setups_outside_cut < max_setups)) continue;
    setups_outside_cut += outside;
    out.moves.push_back(ranked[i]);
    out.equity_ranks.push_back(int32_t(i));
    out.highlighted.push_back(setup);
  }
  if (setups_outside_cut == 0) return {};
  return out;
}

std::vector<SimmedPosition> run_job(const std::vector<char>& buf, const Dictionary& dict,
                                    const SlogSimConfig& config,
                                    const std::vector<binlog::GamePositionIndex>& work,
                                    util::ProgressMeter* meter, bool run_sims) {
  SimJob job{buf.data(), dict, config, work, run_sims, {}, std::vector<SimmedPosition>(work.size()),
             meter};
  const int threads = std::clamp<int>(config.threads, 1, std::max<size_t>(1, work.size()));
  std::vector<std::exception_ptr> errors(threads);
  std::vector<std::thread> workers;
  for (int t = 0; t < threads; ++t) workers.emplace_back(position_worker, &job, &errors[t]);
  for (std::thread& w : workers) w.join();
  for (const std::exception_ptr& e : errors) {
    if (e) std::rethrow_exception(e);
  }
  return std::move(job.results);
}

}  // namespace

SimCandidateSelector setup_selector(const Dictionary& dict, int cut, int max_setups) {
  return [&dict, cut, max_setups](const SimPosition& pos, const std::vector<Move>& ranked,
                                  const Move&, std::mt19937_64&) {
    return select_setup_candidates(pos, dict, ranked, cut, max_setups);
  };
}

std::vector<SimmedPosition> sim_slog_positions(const std::vector<char>& buf, const Dictionary& dict,
                                               const SlogSimConfig& config,
                                               const std::vector<binlog::GamePositionIndex>& work,
                                               util::ProgressMeter* meter) {
  return run_job(buf, dict, config, work, meter, /*run_sims=*/true);
}

std::vector<SimmedPosition> select_slog_candidates(
  const std::vector<char>& buf, const Dictionary& dict, const SlogSimConfig& config,
  const std::vector<binlog::GamePositionIndex>& work, util::ProgressMeter* meter) {
  return run_job(buf, dict, config, work, meter, /*run_sims=*/false);
}

}  // namespace scribblez
