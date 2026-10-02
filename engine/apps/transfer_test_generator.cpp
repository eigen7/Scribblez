// The data generator for SupremeBot M1a's held-out transfer test
// (docs/plans/supreme_bot_m1a.md). Its one mode so far is step 0's
// measurement, which sets the corpus size before any corpus is built.
//
// --mode=measure: for each sampled position of each face-up .slog, select the
// transfer test's candidates (sim/transfer_candidates.h) and sim them with the
// labeling estimator: value-truncated HastyBot rollouts scored by the leaf
// model, under common random numbers. It writes two sidecars per .slog:
//
//   <stem>.trollouts   every rollout's expected score (W + D/2) and final
//                         delta, as float32 pairs: per position, candidate by
//                         candidate, rollout by rollout
//   <stem>.tmeasure  the run's parameters and timings, and per position
//                         its candidates (rank, equity, stratum), the coupled
//                         pairs among them, the pairs the position offered,
//                         the ply-one option saturation curves, and where its
//                         block of the .f32 file starts
//
// Option saturation: on each candidate's post-move board, the union of the
// opponent's static-equity top `option-k` over the racks the first
// `saturation-probes` rollouts dealt them, recorded at doubling probe counts.

#include "agent/agent.h"
#include "data/binary_log.h"
#include "data/gcg_writer.h"
#include "data/slog_sampling.h"
#include "lexicon/dictionary.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "nn/trt_eval_service.h"
#include "nn/trt_util.h"
#include "sim/sim_runner.h"
#include "sim/slog_position_simmer.h"
#include "sim/transfer_candidates.h"
#include "util/exception.h"
#include "util/misc.h"
#include "util/progress.h"

#include <boost/json.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace json = boost::json;
using namespace scribblez;

constexpr int kVersion = 1;
constexpr const char* kJsonExt = ".tmeasure";
constexpr const char* kFloatsExt = ".trollouts";

struct Options {
  std::string mode;
  std::string slog_dir;
  std::vector<std::string> slog_files;
  std::string leaf_model;
  int horizon = 3;
  int rollouts = 10000;
  int positions_per_game = 1;
  int saturation_probes = 256;
  int option_k = 16;
  int threads = util::default_thread_count();
  uint64_t seed = 0;
  int limit_games = 0;
};

const TransferRecipe kRecipe{};

void validate(const Options& opt) {
  if (opt.mode != "measure") throw util::CleanException("--mode must be measure");
  SimRunner::validate_horizon("transfer-test-generator", opt.horizon, !opt.leaf_model.empty());
  if (opt.horizon <= 0) throw util::CleanException("--horizon must be > 0: labels are truncated");
  if (opt.rollouts < 1) throw util::CleanException("--rollouts must be >= 1");
  if (opt.saturation_probes < 1 || opt.saturation_probes > opt.rollouts)
    throw util::CleanException("--saturation-probes must be in [1, --rollouts]");
  if (opt.option_k < 1) throw util::CleanException("--option-k must be >= 1");
}

SlogSimConfig sim_config(const Options& opt, nn::PositionEvalService* leaf) {
  SlogSimConfig c;
  c.open_leaves = true;
  c.selector = transfer_selector(kRecipe);
  c.runner.rollouts = opt.rollouts;
  // A file holds fewer positions than a machine has cores, and their costs
  // vary, so the threads go to one position's rollouts, not across positions.
  c.runner.threads = opt.threads;
  c.runner.horizon_plies = opt.horizon;
  c.runner.leaf_service = leaf;
  c.seed = opt.seed;
  c.threads = 1;
  c.output = SimOutput::kRollouts;
  return c;
}

// The bag after the candidate and the opponent's refill: what the opponent's
// reply sees. An exchange's tiles go back after both refills (Game::play_from).
int bag_after_refills(const SimmedPosition& r, const Move& m) {
  int bag = r.bag_size;
  if (m.type() == MoveType::PLAY) bag -= std::min(bag, m.num_glyphs());
  return bag - std::min(bag, RACK_SIZE - r.position.opp_leave.size());
}

Rack leave_after(const SimPosition& pos, const Move& m) {
  Rack leave = pos.rack;
  for (int i = 0; i < m.num_glyphs(); ++i) leave.remove(m.glyph(i).rack_tile());
  return leave;
}

using MoveKey = std::array<char, sizeof(Move)>;

MoveKey key_of(const Move& m) {
  MoveKey k;
  std::memcpy(k.data(), &m, sizeof(Move));
  return k;
}

// The union's size after 1, 2, 4, ... probes, and after the last.
json::array saturation_curve(const SimmedPosition& r, const Dictionary& dict, int c,
                             const Options& opt) {
  const Move& m = r.candidates.moves[c];
  Board board = r.position.board;
  if (m.type() == MoveType::PLAY) board.apply(m);
  const Rack ours = leave_after(r.position, m);
  const int mover = r.position.mover;
  const int our_score = r.position.scores[mover] + (m.type() == MoveType::PLAY ? m.score() : 0);
  const int bag = bag_after_refills(r, m);
  std::set<MoveKey> seen;
  json::array curve;
  for (int i = 0; i < opt.saturation_probes; ++i) {
    const Rack& theirs = r.rollouts[c][i].opp_rack;
    const MoveRequest req{board, dict, theirs, ours, r.position.scores[1 - mover], our_score, bag};
    for (const Move& o : equity_top_k(req, opt.option_k)) seen.insert(key_of(o));
    const int n = i + 1;
    if ((n & (n - 1)) == 0 || n == opt.saturation_probes) curve.push_back({n, seen.size()});
  }
  return curve;
}

const char* stratum_name(Stratum s) {
  switch (s) {
    case Stratum::kTop:
      return "top";
    case Stratum::kMiddle:
      return "middle";
    case Stratum::kExchange:
      return "exchange";
    case Stratum::kLow:
      return "low";
  }
  return "";
}

const char* coupling_name(Coupling k) {
  switch (k) {
    case Coupling::kPlayExchange:
      return "play_exchange";
    case Coupling::kSameTiles:
      return "same_tiles";
    case Coupling::kSameLaneOneTile:
      return "same_lane_one_tile";
    case Coupling::kNone:
      break;
  }
  return "none";
}

json::array candidates_json(const SimmedPosition& r) {
  json::array out;
  for (size_t c = 0; c < r.candidates.moves.size(); ++c) {
    const Move& m = r.candidates.moves[c];
    const int rank = r.candidates.equity_ranks[c];
    out.push_back({{"move", spelled_move_notation(r.position.board, m)},
                   {"rank", rank},
                   {"equity", r.candidates.equities[c]},
                   {"stratum", stratum_name(stratum_of(m, rank, kRecipe))}});
  }
  return out;
}

json::array couplings_json(const SimmedPosition& r) {
  json::array out;
  for (const CoupledPair& p : find_couplings(r.candidates.moves))
    out.push_back({{"a", p.a}, {"b", p.b}, {"kind", coupling_name(p.kind)}});
  return out;
}

// The coupled pairs the position offered, by kind: what selection drew from.
json::object offered_json(const SimmedPosition& r, const Dictionary& dict) {
  const SimPosition& pos = r.position;
  const MoveRequest req{
    pos.board, dict, pos.rack, pos.opp_leave, pos.scores[pos.mover], pos.scores[1 - pos.mover],
    r.bag_size};
  const AnchoredCouplings pairs =
    anchored_couplings(equity_top_k(req, std::numeric_limits<int>::max()), kRecipe);
  json::object out;
  for (int k = 0; k < kCouplingKinds; ++k) out[coupling_name(Coupling(k + 1))] = pairs[k].size();
  return out;
}

// Every candidate's saturation curve, parallel to the candidates, each on a
// thread of its own: the costly part of a position's record.
json::array saturation_curves(const SimmedPosition& r, const Dictionary& dict, const Options& opt) {
  std::vector<json::array> curves(r.candidates.moves.size());
  std::vector<std::thread> workers;
  for (size_t c = 0; c < curves.size(); ++c)
    workers.emplace_back([&, c] { curves[c] = saturation_curve(r, dict, int(c), opt); });
  for (std::thread& t : workers) t.join();
  json::array out;
  for (json::array& curve : curves) out.push_back(std::move(curve));
  return out;
}

// One position's measurement record.
json::object position_json(const SimmedPosition& r, const Dictionary& dict, const Options& opt,
                           uint64_t float_offset) {
  return {{"game", r.pos.game_idx},
          {"turn", r.pos.turn_idx},
          {"bag_size", r.bag_size},
          {"unseen", r.unseen},
          {"num_legal_moves", r.candidates.num_legal_moves},
          {"played", spelled_move_notation(r.position.board, r.played)},
          {"float_offset", float_offset},
          {"candidates", candidates_json(r)},
          {"couplings", couplings_json(r)},
          {"offered_couplings", offered_json(r, dict)},
          {"saturation", saturation_curves(r, dict, opt)}};
}

void append_rollouts(const SimmedPosition& r, std::vector<float>* floats) {
  for (const std::vector<Rollout>& rollouts : r.rollouts) {
    for (const Rollout& o : rollouts) {
      floats->push_back(float(o.p_win + 0.5 * o.p_draw));
      floats->push_back(float(o.delta));
    }
  }
}

void write_atomically(const fs::path& path, const char* data, size_t size) {
  fs::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary);
    if (!out) throw util::CleanException("cannot write {}", tmp.string());
    out.write(data, std::streamsize(size));
  }
  fs::rename(tmp, path);
}

json::object header_json(const Options& opt, const std::string& leaf_hash) {
  return {{"version", kVersion},
          {"mode", opt.mode},
          {"rollouts", opt.rollouts},
          {"horizon", opt.horizon},
          {"leaf_model_hash", leaf_hash},
          {"seed", opt.seed},
          {"positions_per_game", opt.positions_per_game},
          {"saturation_probes", opt.saturation_probes},
          {"option_k", opt.option_k},
          {"recipe",
           {{"top", kRecipe.top},
            {"middle", kRecipe.middle},
            {"exchanges", kRecipe.exchanges},
            {"low", kRecipe.low},
            {"top_ranks", kRecipe.top_ranks},
            {"middle_ranks", kRecipe.middle_ranks}}}};
}

std::vector<binlog::GamePositionIndex> sampled_work(const std::vector<char>& buf,
                                                    const Options& opt) {
  const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(buf.data());
  const auto* metas =
    reinterpret_cast<const binlog::GameMetadata*>(buf.data() + sizeof(binlog::FileHeader));
  uint32_t games = hdr->num_games;
  if (opt.limit_games > 0) games = std::min<uint32_t>(games, opt.limit_games);
  std::vector<binlog::GamePositionIndex> work;
  for (uint32_t g = 0; g < games; ++g)
    binlog::sample_eligible_turns(metas[g], g, opt.seed, opt.positions_per_game, &work);
  std::ranges::sort(work);
  return work;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Sim and measure one .slog's positions one at a time, so only one position's
// unreduced rollouts are ever held, and write its two sidecars.
void process_file(const binlog::PendingSlog& slog, const Dictionary& dict, const Options& opt,
                  nn::PositionEvalService* leaf, const std::string& leaf_hash,
                  util::ProgressMeter* meter) {
  const std::vector<binlog::GamePositionIndex> work = sampled_work(slog.bytes, opt);
  const SlogSimConfig config = sim_config(opt, leaf);
  json::array positions;
  std::vector<float> floats;
  double sim_s = 0, measure_s = 0;
  for (const binlog::GamePositionIndex& at : work) {
    const auto t0 = std::chrono::steady_clock::now();
    const SimmedPosition r =
      std::move(sim_slog_positions(slog.bytes, dict, config, {at}, meter)[0]);
    sim_s += seconds_since(t0);
    if (r.candidates.moves.empty()) continue;
    const auto t1 = std::chrono::steady_clock::now();
    positions.push_back(position_json(r, dict, opt, floats.size()));
    measure_s += seconds_since(t1);
    append_rollouts(r, &floats);
  }
  json::object doc = header_json(opt, leaf_hash);
  doc["sim_seconds"] = sim_s;
  doc["measure_seconds"] = measure_s;
  doc["threads"] = opt.threads;
  doc["positions"] = std::move(positions);
  write_atomically(slog.sidecar(kFloatsExt), reinterpret_cast<const char*>(floats.data()),
                   floats.size() * sizeof(float));
  const std::string text = json::serialize(doc);
  write_atomically(slog.sidecar(kJsonExt), text.data(), text.size());
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    Options opt;
    po::options_description desc("transfer_test_generator options");
    desc.add_options()("help,h", "show this help and exit")(
      "mode", po::value<std::string>(&opt.mode)->required(), "what to generate: measure")(
      "slog-dir", po::value<std::string>(&opt.slog_dir),
      "directory of face-up .slog files; each without a .tmeasure gets one")(
      "slog-file", po::value<std::vector<std::string>>(&opt.slog_files),
      "explicit .slog file to process (repeatable; overrides --slog-dir)")(
      "leaf-model", po::value<std::string>(&opt.leaf_model)->required(),
      "position evaluation model (.onnx) scoring rollout horizons")(
      "horizon", po::value<int>(&opt.horizon)->default_value(opt.horizon),
      "plies before the leaf model scores a rollout")(
      "rollouts", po::value<int>(&opt.rollouts)->default_value(opt.rollouts),
      "rollouts per candidate")(
      "positions-per-game",
      po::value<int>(&opt.positions_per_game)->default_value(opt.positions_per_game),
      "eligible turns sampled per game")(
      "saturation-probes",
      po::value<int>(&opt.saturation_probes)->default_value(opt.saturation_probes),
      "rollouts whose opponent racks feed the option saturation curves")(
      "option-k", po::value<int>(&opt.option_k)->default_value(opt.option_k),
      "options recorded per rack: its static-equity top k")(
      "threads", po::value<int>(&opt.threads)->default_value(opt.threads), "parallel workers")(
      "seed", po::value<uint64_t>(&opt.seed)->default_value(opt.seed),
      "run seed (drives position sampling, selection and rollout seeds)")(
      "limit-games", po::value<int>(&opt.limit_games)->default_value(opt.limit_games),
      "process only the first N games of each file (0 = all); for smoke runs");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);
    validate(opt);
    const Dictionary& dict = load_dictionary_or_throw();
    HastyEquity::ensure_initialized(Lexicon::instance().name());
    const std::vector<binlog::PendingSlog> pending = binlog::load_pending_slogs(
      binlog::resolve_slog_inputs(opt.slog_dir, opt.slog_files), kJsonExt,
      /*accept_face_up=*/true, "");
    for (const binlog::PendingSlog& p : pending) {
      const auto* hdr = reinterpret_cast<const binlog::FileHeader*>(p.bytes.data());
      if (!(hdr->flags & binlog::kFlagFaceUpLeaves))
        throw util::CleanException("{} was not played with face-up leaves", p.path.string());
    }
    if (pending.empty()) return 0;
    uint64_t total = 0;
    for (const binlog::PendingSlog& p : pending)
      total += binlog::count_sampled_positions(p.bytes, opt.positions_per_game, opt.limit_games);
    std::cerr << "transfer-test: " << pending.size() << " file(s), " << total << " positions; "
              << kRecipe.size() << " candidates x " << opt.rollouts << " rollouts, " << opt.threads
              << " threads\n";
    const std::shared_ptr<nn::PositionEvalService> leaf =
      nn::load_leaf_position_service(opt.leaf_model);
    const std::string leaf_hash = nn::content_hash(binlog::read_file_bytes(opt.leaf_model));
    util::ProgressMeter meter(total, "positions");
    for (const binlog::PendingSlog& p : pending)
      process_file(p, dict, opt, leaf.get(), leaf_hash, &meter);
    meter.finish("transfer-test");
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
