#include "sim/slog_position_simmer.h"

#include "agent/agent.h"
#include "data/binary_log.h"
#include "encoding/position_encoder.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "sim/setup_plays.h"
#include "util/exception.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <numeric>
#include <thread>

namespace scribblez {

namespace {

int32_t equity_rank(const std::vector<Move>& ranked, const Move& m) {
  const auto it = std::find(ranked.begin(), ranked.end(), m);
  return it == ranked.end() ? -1 : int32_t(it - ranked.begin());
}

SimPosition sim_position(const binlog::PositionEncoder& encoder, const GameLog& g, int turn_idx,
                         int mover, bool open_leaves) {
  SimPosition pos;
  pos.board = encoder.enc().board();
  pos.scores = {encoder.enc().score(0), encoder.enc().score(1)};
  pos.mover = mover;
  pos.rack = encoder.rack(mover);
  // The replay knows the opponent's rack and the draws after their last move,
  // so it can recover exactly which tiles they kept.
  if (open_leaves)
    pos.opp_leave = binlog::opp_leave_from_replay(g, turn_idx, encoder.rack(1 - mover));
  return pos;
}

// Every legal move at `pos`, best static equity first. The ranking may see only
// what the mover legitimately knows: the opponent's kept tiles under face-up
// leaves, never their replayed rack.
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

// Each candidate's rollouts, in rollout-index order.
using CandidateRollouts = std::vector<std::vector<RolloutResult>>;

// Append rollouts [done, upto) of the `alive` candidates.
void run_instalment(const SimRunner& runner, const SimmedPosition& res,
                    const std::vector<size_t>& alive, int done, int upto,
                    CandidateRollouts* rollouts) {
  std::vector<Move> moves;
  for (const size_t c : alive) moves.push_back(res.candidates.moves[c]);
  const int count = upto - done;
  const std::vector<RolloutResult> flat =
    runner.run_rollouts(res.position, moves, res.base_seed + uint64_t(done), count);
  for (size_t k = 0; k < alive.size(); ++k)
    (*rollouts)[alive[k]].insert((*rollouts)[alive[k]].end(), flat.begin() + k * size_t(count),
                                 flat.begin() + (k + 1) * size_t(count));
}

// The alive candidate with the best win rate so far.
size_t race_leader(const std::vector<size_t>& alive, const CandidateRollouts& rollouts) {
  size_t leader = alive.front();
  double best = -1;
  for (const size_t c : alive) {
    double wins = 0;
    for (const RolloutResult& r : rollouts[c]) wins += r.p_win + 0.5 * r.p_draw;
    if (wins > best) {
      best = wins;
      leader = c;
    }
  }
  return leader;
}

// Drop the unprotected candidates clearly below the leader (see
// SlogSimConfig::race_checkpoints).
void stop_the_beaten(const SlogSimConfig& config, const CandidateRollouts& rollouts,
                     std::vector<size_t>* alive) {
  const std::vector<RolloutResult>& lead = rollouts[race_leader(*alive, rollouts)];
  std::erase_if(*alive, [&](size_t c) {
    if (c < size_t(config.race_protected)) return false;
    return clearly_below(paired_win_diff(rollouts[c], lead), lead.size(), config.race_sigmas);
  });
}

// Sim the position's candidates to runner.rollouts(), racing them when the
// config sets checkpoints.
CandidateRollouts sim_candidates(const SlogSimConfig& config, const SimRunner& runner,
                                 const SimmedPosition& res) {
  CandidateRollouts rollouts(res.candidates.moves.size());
  std::vector<size_t> alive(rollouts.size());
  std::iota(alive.begin(), alive.end(), size_t(0));
  int done = 0;
  for (const int upto : config.race_checkpoints) {
    run_instalment(runner, res, alive, done, upto, &rollouts);
    done = upto;
    if (done < runner.rollouts()) stop_the_beaten(config, rollouts, &alive);
  }
  if (done < runner.rollouts())
    run_instalment(runner, res, alive, done, runner.rollouts(), &rollouts);
  return rollouts;
}

void summarize_position(const SlogSimConfig& config, const CandidateRollouts& rollouts,
                        SimmedPosition* res) {
  const size_t n = rollouts.size();
  const size_t refs = std::min(n, size_t(config.paired_references));
  res->paired.assign(n, std::vector<PairedWinDiff>(refs));
  for (size_t c = 0; c < n; ++c) {
    res->summaries.push_back(summarize_rollouts(res->candidates.moves[c], rollouts[c]));
    for (size_t r = 0; r < refs; ++r) res->paired[c][r] = paired_win_diff(rollouts[c], rollouts[r]);
  }
}

// Sim the position's candidates and keep what the config asks for.
void reduce_rollouts(const SlogSimConfig& config, const SimRunner& runner, SimmedPosition* res) {
  if (config.keep_summaries) {
    summarize_position(config, sim_candidates(config, runner, *res), res);
  } else {
    res->observations = runner.run(res->position, res->candidates.moves, res->base_seed);
  }
}

// A worker's SimRunners: the config's own, plus one with solve_endgames on
// when the config solves late positions' endgames.
struct PositionRunners {
  PositionRunners(const Dictionary& dict, const SlogSimConfig& config);

  SimRunner standard;
  std::optional<SimRunner> solving;
};

SimRunner::Params with_solved_endgames(SimRunner::Params params) {
  params.solve_endgames = true;
  return params;
}

PositionRunners::PositionRunners(const Dictionary& dict, const SlogSimConfig& config)
    : standard(dict, config.runner) {
  if (config.solve_endgames_max_unseen >= 0)
    solving.emplace(dict, with_solved_endgames(config.runner));
}

void sim_one_position(const binlog::GamePositionIndex& w, SimJob* job,
                      binlog::PositionEncoder* encoder, std::vector<TurnRecord>* scratch,
                      const PositionRunners& runners, SimmedPosition* res) {
  const GameLog g = binlog::make_game_view(job->buf, w.game_idx, *scratch, nullptr);
  const int mover = encoder->replay_to_sampled(g, int(w.turn_idx), /*post_move=*/false);
  const SimPosition pos =
    sim_position(*encoder, g, int(w.turn_idx), mover, job->config.open_leaves);
  res->pos = w;
  res->position = pos;
  const uint64_t position_seed = binlog::position_seed(job->config.seed, w.game_idx, w.turn_idx);
  res->base_seed = position_seed + job->config.rollout_seed_offset;
  std::mt19937_64 rng(position_seed);
  res->bag_size = encoder->bag_size();
  res->played = g.records[w.turn_idx].move;
  res->candidates =
    job->config.selector(w, pos, rank_candidates(pos, job->dict, res->bag_size), res->played, rng);
  if (res->candidates.moves.empty()) return;
  res->candidates.equities = HastyEquity::instance().equities(
    res->candidates.moves, pos.board, res->bag_size, pos.opp_leave, pos.rack);
  res->unseen = pos.board.unseen_tiles(pos.rack).size();
  res->solved_endgames = runners.solving && res->unseen <= job->config.solve_endgames_max_unseen;
  const SimRunner& runner = res->solved_endgames ? *runners.solving : runners.standard;
  if (job->run_sims) reduce_rollouts(job->config, runner, res);
}

// Claims positions off the shared index and fills their result slots.
void run_position_worker(SimJob* job) {
  std::vector<TurnRecord> scratch;
  binlog::PositionEncoder encoder(InputEncodingSpec{&job->dict});
  const PositionRunners runners(job->dict, job->config);
  const size_t n = job->work.size();
  for (size_t i = job->next.fetch_add(1); i < n; i = job->next.fetch_add(1)) {
    const binlog::GamePositionIndex& w = job->work[i];
    // Name the failing position, so an unattended run leaves a lead.
    try {
      sim_one_position(w, job, &encoder, &scratch, runners, &job->results[i]);
    } catch (const std::exception& e) {
      throw util::Exception("game {} turn {}: {}", w.game_idx, w.turn_idx, e.what());
    }
    job->meter->add_done();
  }
}

// Thread entry: captures any exception into *err for the joining thread to
// rethrow, since one escaping a std::thread terminates the process.
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
  return [recipe](const binlog::GamePositionIndex&, const SimPosition&,
                  const std::vector<Move>& ranked, const Move& played, std::mt19937_64& rng) {
    return select_sim_candidates(ranked, played, recipe, rng);
  };
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

SimCandidates select_all_plays(const SimPosition& pos, const Dictionary& dict,
                               const std::vector<Move>& ranked, int cut, int max_plays) {
  SimCandidates out;
  out.num_legal_moves = ranked.size();
  int beyond_cut = 0;
  for (size_t i = 0; i < ranked.size(); ++i) {
    const Move& m = ranked[i];
    if (int(i) >= cut) {
      const bool wanted = m.type() == MoveType::PLAY && !places_blank(m);
      if (!wanted || (max_plays > 0 && beyond_cut >= max_plays)) continue;
      ++beyond_cut;
    }
    out.moves.push_back(m);
    out.equity_ranks.push_back(int32_t(i));
    out.highlighted.push_back(is_high_value_setup(pos.board, dict, pos.rack, m));
  }
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
  return [&dict, cut, max_setups](const binlog::GamePositionIndex&, const SimPosition& pos,
                                  const std::vector<Move>& ranked, const Move&, std::mt19937_64&) {
    return select_setup_candidates(pos, dict, ranked, cut, max_setups);
  };
}

SimCandidateSelector all_plays_selector(const Dictionary& dict, int cut, int max_plays) {
  return [&dict, cut, max_plays](const binlog::GamePositionIndex&, const SimPosition& pos,
                                 const std::vector<Move>& ranked, const Move&, std::mt19937_64&) {
    return select_all_plays(pos, dict, ranked, cut, max_plays);
  };
}

SimCandidateSelector chosen_selector(ChosenMoves chosen) {
  return
    [chosen = std::move(chosen)](const binlog::GamePositionIndex& at, const SimPosition&,
                                 const std::vector<Move>& ranked, const Move&, std::mt19937_64&) {
      SimCandidates out;
      const auto it = chosen.find(at);
      if (it == chosen.end()) return out;
      out.num_legal_moves = ranked.size();
      out.moves = it->second;
      for (const Move& m : out.moves) out.equity_ranks.push_back(equity_rank(ranked, m));
      out.highlighted.assign(out.moves.size(), 0);
      return out;
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
