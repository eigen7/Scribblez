#include "serve/scribblez_ffi.h"

#include "agent/agent.h"
#include "agent/player_factory.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"
#include "data/gcg_reader.h"
#include "data/slog_subset.h"
#include "data/streaming_row_buffer.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "encoding/row_encoder.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "selfplay/self_play_engine.h"
#include "selfplay/sim_observation_log.h"
#include "selfplay/sim_runner.h"
#include "selfplay/streaming_game_producer.h"
#include "training/lane_analysis.h"
#include "training/lane_targets.h"
#include "training/max_move_per_lane_input_encoder.h"
#include "training/max_move_per_lane_task.h"
#include "training/move_set_encoder.h"
#include "training/position_eval_analysis.h"
#include "training/training_targets.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using scribblez::binlog::DataLoader;
using scribblez::binlog::FileHeader;
using scribblez::binlog::kMagic;
using scribblez::binlog::kVersion;

// The session behind the C ABI (see scribblez_ffi.h). Owns the process's
// InputEncodingSpec -- the loaded Dictionary plus the feature-block choice --
// that every encoding / analysis / loader entry point derives from, and the
// spec's input tensor shapes, which the shape/size queries report. A
// constructed session is proof the lexicon is loaded, so the methods do no
// load-failure checking.
struct ScribblezSession {
  ScribblezSession(const char* lexicon_name, bool contingent_features, bool opp_leave_input);

  const ScribblezShape* input_shapes() const { return input_shapes_; }
  int input_floats() const { return scribblez::input_floats(spec); }
  int row_size_floats() const { return input_floats() + scribblez::kLabelFloats; }

  int encode_score_diff_sweep(const char* path, int64_t game_idx, bool post_move, int diff_lo,
                              int diff_hi, float* out_inputs) const;
  int decode_rows(const char* path, const int64_t* game_idx, const int64_t* turn_idx, int64_t n,
                  bool post_move, float* out) const;
  int gcg_sim_evidence(const char* gcg_text, int top_k, int rollouts, int threads, uint64_t seed,
                       bool open_leaves, char* out_records, int* played_rank) const;
  int dump_position(const char* path, int64_t game_idx, bool post_move, char* out,
                    int out_cap) const;
  int dump_position_json(const char* path, int64_t game_idx, bool post_move, char* out,
                         int out_cap) const;
  int max_move_per_lane_analyze_gcg(const char* gcg_text, char* out_json, int out_cap,
                                    float* out_input) const;
  int position_eval_analyze_gcg(const char* gcg_text, float* out_input) const;
  int position_eval_analyze_gcg_leave(const char* gcg_text, const char* leave_str, float* out_input,
                                      char* out_err, int err_cap) const;
  DataLoaderHandle* dl_new(int64_t memory_budget, int num_worker_threads, int num_prefetch_threads,
                           int task) const;
  StreamHandle* stream_new(float* const* slot_ptrs, int num_slots, int rows_per_slot,
                           int num_threads, bool post_move, bool apply_symmetry, uint64_t seed,
                           int handicap_max, const char* const* player_specs, int num_specs) const;
  StreamHandle* max_move_per_lane_stream_new(float* const* slot_ptrs, int num_slots,
                                             int rows_per_slot, int num_threads,
                                             bool apply_symmetry, uint64_t seed, int handicap_max,
                                             const char* const* player_specs, int num_specs) const;

  scribblez::InputEncodingSpec spec;

 private:
  // The spec's input tensor shapes, advertised through input_shapes(). The dim
  // arrays live in the session so the pointers stay valid for its lifetime.
  int spatial_dims_[3];
  int scalar_dims_[1];
  ScribblezShape input_shapes_[3];
};

namespace {

// Throws when the .kwg is missing, having printed an install hint; uncaught
// across the C ABI, that terminates the process.
const scribblez::Dictionary& load_session_dictionary(const char* lexicon_name) {
  scribblez::Lexicon& lex = scribblez::Lexicon::instance();
  lex.set_params({.name = lexicon_name, .dir = lex.dir()});
  return scribblez::load_dictionary_or_throw();
}

}  // namespace

ScribblezSession::ScribblezSession(const char* lexicon_name, bool contingent_features,
                                   bool opp_leave_input)
    : spec{&load_session_dictionary(lexicon_name), contingent_features, opp_leave_input},
      spatial_dims_{scribblez::spatial_planes(spec), scribblez::kBoardSide, scribblez::kBoardSide},
      scalar_dims_{scribblez::scalar_floats(spec)},
      input_shapes_{{"input_spatial", spatial_dims_, 3, -1},
                    {"input_scalar", scalar_dims_, 1, -1},
                    {nullptr, nullptr, 0, 0}} {}

ScribblezSession* scribblez_session_new(const char* lexicon_name, int contingent_features,
                                        int opp_leave_input) {
  return new ScribblezSession(lexicon_name, contingent_features != 0, opp_leave_input != 0);
}

void scribblez_session_delete(ScribblezSession* s) { delete s; }

namespace {

// Build the (null-terminated) target shape table at compile time directly
// from scribblez::AllTargets, so adding/removing a target struct
// in training_targets.h automatically updates the FFI advertisement with
// no edits here.
template <typename List>
struct TargetShapeTable;

template <typename... Ts>
struct TargetShapeTable<scribblez::TargetList<Ts...>> {
  static constexpr std::size_t kCount = sizeof...(Ts);
  static constexpr std::array<ScribblezShape, kCount + 1> kValue = []() {
    std::array<ScribblezShape, kCount + 1> a{};
    std::size_t i = 0;
    (void)std::initializer_list<int>{
      (a[i] = ScribblezShape{Ts::kName, Ts::kDims, static_cast<int>(std::size(Ts::kDims)),
                             static_cast<int>(i)},
       ++i, 0)...};
    a[kCount] = ScribblezShape{nullptr, nullptr, 0, 0};
    return a;
  }();
};

constexpr auto kTargetShapesArr = TargetShapeTable<scribblez::AllTargets>::kValue;

// --- Max-move-per-lane task shapes (hand-written; the max-move-per-lane encoders are not part of
// the AllTargets pack). Input: 31 board planes + 27 rack scalars. Labels: the
// per-lane occupancy / score / mask blocks of encode_lane_targets. ---
constexpr int kMaxMovePerLaneInputSpatialDims[3] = {
  scribblez::MaxMovePerLaneInputEncoder::kSpatialPlanes, scribblez::kBoardSide,
  scribblez::kBoardSide};
constexpr int kMaxMovePerLaneInputScalarDims[1] = {
  scribblez::MaxMovePerLaneInputEncoder::kScalarFloats};

const ScribblezShape kMaxMovePerLaneInputShapes[] = {
  {"input_spatial", kMaxMovePerLaneInputSpatialDims, 3, -1},
  {"input_scalar", kMaxMovePerLaneInputScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

constexpr int kMaxMovePerLaneOccupancyDims[3] = {scribblez::kNumLanes, scribblez::kLaneLen,
                                                 scribblez::kLaneTileKinds};
constexpr int kMaxMovePerLaneScoreDims[1] = {scribblez::kNumLanes};
constexpr int kMaxMovePerLaneMaskDims[1] = {scribblez::kNumLanes};

const ScribblezShape kMaxMovePerLaneTargetShapes[] = {
  {"lane_occupancy", kMaxMovePerLaneOccupancyDims, 3, 0},
  {"lane_score", kMaxMovePerLaneScoreDims, 1, 1},
  {"lane_mask", kMaxMovePerLaneMaskDims, 1, 2},
  {nullptr, nullptr, 0, 0},
};

}  // namespace

extern "C" {

const ScribblezShape* scribblez_input_shapes(ScribblezSession* s) { return s->input_shapes(); }
const ScribblezShape* scribblez_target_shapes(void) { return kTargetShapesArr.data(); }

int scribblez_row_size_floats(ScribblezSession* s) { return s->row_size_floats(); }

int scribblez_input_floats(ScribblezSession* s) { return s->input_floats(); }

const ScribblezShape* scribblez_max_move_per_lane_input_shapes(void) {
  return kMaxMovePerLaneInputShapes;
}
const ScribblezShape* scribblez_max_move_per_lane_target_shapes(void) {
  return kMaxMovePerLaneTargetShapes;
}

int scribblez_max_move_per_lane_row_size_floats(void) {
  return scribblez::MaxMovePerLaneTask::kRowFloats;
}

int scribblez_max_move_per_lane_input_floats(void) {
  return scribblez::MaxMovePerLaneInputEncoder::kInputFloats;
}

namespace {

// Read a .slog file into `buf`, validate its header, and report the game
// count, bounds-checking `game_idx` when it is >= 0. Returns 0 on
// success, -1 on any failure.
int load_slog(const char* path, int64_t game_idx, std::vector<char>& buf, uint32_t* num_games) {
  if (!path) return -1;
  std::ifstream f(path, std::ios::binary);
  if (!f) return -1;
  buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < sizeof(FileHeader)) return -1;
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  if (hdr->magic != kMagic || hdr->version != kVersion) return -1;
  if (game_idx >= static_cast<int64_t>(hdr->num_games)) return -1;
  if (num_games) *num_games = hdr->num_games;
  return 0;
}

}  // namespace

int ScribblezSession::encode_score_diff_sweep(const char* path, int64_t game_idx, bool post_move,
                                              int diff_lo, int diff_hi, float* out_inputs) const {
  if (!out_inputs || diff_hi < diff_lo) return -1;
  uint32_t num_games = 0;
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, &num_games) != 0) return -1;

  const int64_t sweep = static_cast<int64_t>(diff_hi - diff_lo + 1) * input_floats();
  scribblez::binlog::BlockDecoder decoder(spec);
  if (game_idx >= 0) {
    // A single position.
    decoder.encode_score_diff_sweep(buf.data(), static_cast<uint32_t>(game_idx), post_move, diff_lo,
                                    diff_hi, out_inputs);
  } else {
    // Every position in the file, position-major (game g at row g * R).
    for (uint32_t g = 0; g < num_games; ++g) {
      decoder.encode_score_diff_sweep(buf.data(), g, post_move, diff_lo, diff_hi,
                                      out_inputs + static_cast<int64_t>(g) * sweep);
    }
  }
  return 0;
}

int scribblez_encode_score_diff_sweep(ScribblezSession* s, const char* path, int64_t game_idx,
                                      int post_move, int diff_lo, int diff_hi, float* out_inputs) {
  return s->encode_score_diff_sweep(path, game_idx, post_move != 0, diff_lo, diff_hi, out_inputs);
}

namespace {

// Copy `s` into the caller's NUL-terminated buffer (truncating to out_cap) and
// return the full length, which may exceed out_cap - 1 (caller should retry).
int emit_string(const std::string& s, char* out, int out_cap) {
  const int len = static_cast<int>(s.size());
  if (out && out_cap > 0) {
    const int n = len < out_cap - 1 ? len : out_cap - 1;
    std::memcpy(out, s.data(), static_cast<size_t>(n));
    out[n] = '\0';
  }
  return len;
}

}  // namespace

int ScribblezSession::gcg_sim_evidence(const char* gcg_text, int top_k, int rollouts, int threads,
                                       uint64_t seed, bool open_leaves, char* out_records,
                                       int* played_rank) const {
  if (!gcg_text || !out_records || top_k < 1) return -1;
  scribblez::ParsedGcgGame game;
  std::string error;
  if (!scribblez::read_gcg_text(gcg_text, &game, &error)) return -1;
  if (game.turns.empty() || game.snapshots.size() != game.turns.size() + 1) return -1;

  // The decision point: the state before the final recorded move, with the
  // mover's full reconstructed rack.
  const size_t last = game.turns.size() - 1;
  const scribblez::TurnRecord& final_turn = game.turns[last].record;
  scribblez::SimPosition pos;
  pos.board = game.snapshots[last].board;
  pos.scores = game.snapshots[last].scores;
  pos.mover = final_turn.player;
  pos.rack = final_turn.rack_before;

  // Open leaves: the opponent's retained leave is reconstructable from their
  // last recorded move (their replenishments stay hidden and are sampled).
  // An empty leave -- opponent bingoed or has not acted -- is legitimate.
  if (open_leaves) pos.opp_leave = scribblez::retained_leave(game, 1 - pos.mover);

  const int pool_size = scribblez::unseen_pool(pos.board, pos.rack, 0).size();
  if (pool_size <= scribblez::RACK_SIZE) return -1;  // endgame: SimRunner's non-empty-bag rule

  scribblez::HastyEquity::ensure_initialized(scribblez::Lexicon::instance().name());
  // Bag size from the mover's POV: the unseen pool minus the opponent's
  // (assumed full) rack. Only equity's endgame adjustments read the opponent
  // rack, for which open-leaves supplies the known part.
  const scribblez::Rack hidden_opp;
  scribblez::MoveRequest req{pos.board,
                             *spec.dict,
                             pos.rack,
                             open_leaves ? pos.opp_leave : hidden_opp,
                             pos.scores[pos.mover],
                             pos.scores[1 - pos.mover],
                             std::max(0, pool_size - scribblez::RACK_SIZE)};
  const std::vector<scribblez::Move> candidates = scribblez::equity_top_k(req, top_k);

  scribblez::SimRunner::Params params;
  params.rollouts = rollouts;
  params.threads = threads;
  const scribblez::SimRunner runner(*spec.dict, params);
  const std::vector<scribblez::SimObservation> obs = runner.run(pos, candidates, seed);

  *played_rank = -1;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i] == final_turn.move) *played_rank = static_cast<int>(i);
    char* rec = out_records + i * sizeof(scribblez::SimObsRecord);
    std::memcpy(rec, &candidates[i], sizeof(scribblez::Move));
    std::memcpy(rec + sizeof(scribblez::Move), &obs[i], sizeof(scribblez::SimObservation));
  }
  return static_cast<int>(candidates.size());
}

int scribblez_gcg_sim_evidence(ScribblezSession* s, const char* gcg_text, int top_k, int rollouts,
                               int threads, uint64_t seed, int open_leaves, char* out_records,
                               int* played_rank) {
  return s->gcg_sim_evidence(gcg_text, top_k, rollouts, threads, seed, open_leaves != 0,
                             out_records, played_rank);
}

int ScribblezSession::decode_rows(const char* path, const int64_t* game_idx,
                                  const int64_t* turn_idx, int64_t n, bool post_move,
                                  float* out) const {
  if (!game_idx || !turn_idx || !out || n < 0) return -1;
  std::vector<char> buf;
  if (load_slog(path, /*game_idx=*/0, buf, nullptr) != 0) return -1;
  scribblez::binlog::BlockDecoder decoder(spec);
  const int64_t row_floats = scribblez::input_floats(spec) + scribblez::kLabelFloats;
  for (int64_t j = 0; j < n; ++j) {
    decoder.decode_one(buf.data(), path, static_cast<uint32_t>(game_idx[j]),
                       static_cast<uint32_t>(turn_idx[j]), /*flip=*/false, post_move,
                       /*output_row=*/0, out + j * row_floats);
  }
  return 0;
}

int scribblez_decode_rows(ScribblezSession* s, const char* path, const int64_t* game_idx,
                          const int64_t* turn_idx, int64_t n, int post_move, float* out) {
  return s->decode_rows(path, game_idx, turn_idx, n, post_move != 0, out);
}

void scribblez_move_set_encode_moves(const void* moves, int64_t n,
                                     const int32_t* pre_move_score_diffs, int32_t* out_letters,
                                     uint8_t* out_blanks, int32_t* out_squares,
                                     uint8_t* out_tile_mask, float* out_scalars) {
  namespace mset = scribblez::move_set;
  const char* bytes = static_cast<const char*>(moves);
  // The input is a packed byte buffer of serialized Moves whose alignment the
  // caller does not guarantee, so copy each into a Move before encoding.
  for (int64_t i = 0; i < n; ++i) {
    scribblez::Move m;
    std::memcpy(&m, bytes + i * sizeof(scribblez::Move), sizeof(scribblez::Move));
    mset::encode_move(m, pre_move_score_diffs[i], out_letters + i * mset::kMoveMaxPlaced,
                      out_blanks + i * mset::kMoveMaxPlaced, out_squares + i * mset::kMoveMaxPlaced,
                      out_tile_mask + i * mset::kMoveMaxPlaced,
                      out_scalars + i * mset::kMoveScalars);
  }
}

void scribblez_move_set_move_dims(int32_t* max_placed, int32_t* num_scalars, int32_t* letter_vocab,
                                  int32_t* cells) {
  *max_placed = scribblez::move_set::kMoveMaxPlaced;
  *num_scalars = scribblez::move_set::kMoveScalars;
  *letter_vocab = scribblez::move_set::kMoveLetterVocab;
  *cells = scribblez::move_set::kMoveCells;
}

int32_t scribblez_move_set_encoding_version(void) {
  return scribblez::move_set::kMoveEncodingVersion;
}

void scribblez_score_diff_input_layout(ScribblezSession* s, int32_t* scalar_index, float* scale) {
  *scalar_index = scribblez::scalar_block_offset(s->spec, scribblez::ScalarBlockId::kScoreDiff);
  *scale = scribblez::kScoreDiffInputScale;
}

int ScribblezSession::dump_position(const char* path, int64_t game_idx, bool post_move, char* out,
                                    int out_cap) const {
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, nullptr) != 0) return -1;
  scribblez::binlog::BlockDecoder decoder(spec);
  return emit_string(decoder.dump_position(buf.data(), static_cast<uint32_t>(game_idx), post_move),
                     out, out_cap);
}

int scribblez_dump_position(ScribblezSession* s, const char* path, int64_t game_idx, int post_move,
                            char* out, int out_cap) {
  return s->dump_position(path, game_idx, post_move != 0, out, out_cap);
}

int ScribblezSession::dump_position_json(const char* path, int64_t game_idx, bool post_move,
                                         char* out, int out_cap) const {
  std::vector<char> buf;
  if (load_slog(path, game_idx, buf, nullptr) != 0) return -1;
  scribblez::binlog::BlockDecoder decoder(spec);
  return emit_string(
    decoder.dump_position_json(buf.data(), static_cast<uint32_t>(game_idx), post_move), out,
    out_cap);
}

int scribblez_dump_position_json(ScribblezSession* s, const char* path, int64_t game_idx,
                                 int post_move, char* out, int out_cap) {
  return s->dump_position_json(path, game_idx, post_move != 0, out, out_cap);
}

int ScribblezSession::max_move_per_lane_analyze_gcg(const char* gcg_text, char* out_json,
                                                    int out_cap, float* out_input) const {
  if (!gcg_text) return -1;
  try {
    scribblez::GcgAnalysisPosition pos;
    std::string error;
    if (!scribblez::parse_gcg_analysis_position(gcg_text, &pos, &error)) return -1;
    if (out_input)
      scribblez::MaxMovePerLaneInputEncoder::encode(pos.board, pos.rack, /*flip=*/false, out_input);
    return emit_string(scribblez::lane_analysis_json(pos.board, pos.rack, pos.on_move, *spec.dict),
                       out_json, out_cap);
  } catch (const std::exception&) {
    return -1;
  }
}

int scribblez_max_move_per_lane_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                            char* out_json, int out_cap, float* out_input) {
  return s->max_move_per_lane_analyze_gcg(gcg_text, out_json, out_cap, out_input);
}

int ScribblezSession::position_eval_analyze_gcg(const char* gcg_text, float* out_input) const {
  if (!gcg_text || !out_input) return -1;
  try {
    std::string error;
    if (!scribblez::encode_position_eval_analysis_input(gcg_text, spec, out_input, &error)) {
      return -1;
    }
    return input_floats();
  } catch (const std::exception&) {
    return -1;
  }
}

int scribblez_position_eval_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                        float* out_input) {
  return s->position_eval_analyze_gcg(gcg_text, out_input);
}

int scribblez_position_eval_board_json(const char* gcg_text, char* out_json, int out_cap) {
  if (!gcg_text) return -1;
  try {
    std::string error;
    const std::string json = scribblez::position_eval_analysis_board_json(gcg_text, &error);
    if (json.empty()) return -1;
    return emit_string(json, out_json, out_cap);
  } catch (const std::exception&) {
    return -1;
  }
}

int ScribblezSession::position_eval_analyze_gcg_leave(const char* gcg_text, const char* leave_str,
                                                      float* out_input, char* out_err,
                                                      int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !leave_str || !out_input) return -1;
  try {
    std::string error;
    if (!scribblez::encode_position_eval_analysis_input_with_leave(gcg_text, leave_str, spec,
                                                                   out_input, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return input_floats();
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_position_eval_analyze_gcg_leave(ScribblezSession* s, const char* gcg_text,
                                              const char* leave_str, float* out_input,
                                              char* out_err, int err_cap) {
  return s->position_eval_analyze_gcg_leave(gcg_text, leave_str, out_input, out_err, err_cap);
}

int scribblez_sample_slog(const char* dst_path, const char* const* src_paths,
                          const int64_t* game_indices, int num_picks) {
  if (!dst_path || !src_paths || !game_indices || num_picks < 0) return -1;
  std::vector<scribblez::binlog::SlogPick> picks;
  picks.reserve(static_cast<size_t>(num_picks));
  for (int i = 0; i < num_picks; ++i) {
    if (!src_paths[i]) return -1;
    picks.push_back({src_paths[i], game_indices[i]});
  }
  return scribblez::binlog::write_slog_subset(dst_path, picks) ? 0 : -1;
}

int scribblez_read_file_header(const char* path, int64_t* out_num_positions,
                               int64_t* out_file_size) {
  if (!path || !out_num_positions || !out_file_size) return -1;
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return -1;
  FileHeader hdr{};
  const size_t n = std::fread(&hdr, sizeof(hdr), 1, f);
  std::fclose(f);
  if (n != 1) return -1;
  if (hdr.magic != kMagic) return -1;
  if (hdr.version != kVersion) return -1;
  std::error_code ec;
  const auto fsz = std::filesystem::file_size(path, ec);
  if (ec) return -1;
  *out_num_positions = static_cast<int64_t>(hdr.num_games);
  *out_file_size = static_cast<int64_t>(fsz);
  return 0;
}

struct DataLoaderHandle {
  DataLoader loader;
  explicit DataLoaderHandle(const DataLoader::Params& p) : loader(p) {}
};

DataLoaderHandle* ScribblezSession::dl_new(int64_t memory_budget, int num_worker_threads,
                                           int num_prefetch_threads, int task) const {
  DataLoader::Params p;
  p.spec = spec;
  p.task = task == 1 ? scribblez::binlog::DecodeTask::kMaxMovePerLane
                     : scribblez::binlog::DecodeTask::kPositionEval;
  p.memory_budget = memory_budget;
  p.num_worker_threads = num_worker_threads;
  p.num_prefetch_threads = num_prefetch_threads;
  return new DataLoaderHandle(p);
}

DataLoaderHandle* scribblez_dl_new(ScribblezSession* s, int64_t memory_budget,
                                   int num_worker_threads, int num_prefetch_threads, int task) {
  return s->dl_new(memory_budget, num_worker_threads, num_prefetch_threads, task);
}

void scribblez_dl_delete(DataLoaderHandle* h) { delete h; }

void scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_positions,
                           int64_t file_size) {
  if (!h || !path) return;
  h->loader.add_file(path, num_positions, file_size);
}

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.num_positions();
}

int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed, int turns_per_game, int epoch_index) {
  if (!h) return 0;
  DataLoader::EpochConfig cfg;
  cfg.batch_size = batch_size;
  cfg.post_move = post_move != 0;
  cfg.apply_symmetry = apply_symmetry != 0;
  cfg.seed = seed;
  cfg.turns_per_game = turns_per_game;
  cfg.epoch_index = epoch_index;
  return h->loader.epoch_start(cfg);
}

int scribblez_dl_load_batch(DataLoaderHandle* h, float* output) {
  if (!h || !output) return 0;
  return h->loader.load_batch(output);
}

int64_t scribblez_dl_resident_bytes(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.resident_bytes();
}

// ---------------------------------------------------------------------------
// Streaming self-play -> training pipeline
// ---------------------------------------------------------------------------

struct StreamHandle {
  scribblez::binlog::StreamingRowBuffer ring;
  scribblez::binlog::StreamingGameProducer producer;

  StreamHandle(float* const* slots, int num_slots, int rows_per_slot, int row_floats,
               const scribblez::SelfPlayEngine::Params& engine_params,
               const scribblez::PlayerFactory::Params& player_params,
               const scribblez::binlog::StreamingGameProducer::Params& stream_params)
      : ring(slots, num_slots, rows_per_slot, row_floats),
        producer(engine_params, player_params, stream_params, ring) {}
};

namespace {

// Shared construction for both streaming entry points: validate the config,
// build the engine + player params, and create the StreamHandle with the given
// row width and per-worker encoder factory. Returns nullptr on bad config or a
// lexicon-load failure. Both tasks play the same HastyBot self-play games; only
// the per-row encoding (and thus the row width) differs.
StreamHandle* new_stream(float* const* slot_ptrs, int num_slots, int rows_per_slot, int num_threads,
                         int apply_symmetry, uint64_t seed, int handicap_max,
                         const char* const* player_specs, int num_specs, int row_floats,
                         scribblez::binlog::RowEncoderFactory factory) {
  if (!slot_ptrs || num_slots < 1 || rows_per_slot < 1 || !player_specs || num_specs < 1) {
    return nullptr;
  }
  scribblez::PlayerFactory::Params player_params;
  for (int i = 0; i < num_specs; ++i) {
    if (!player_specs[i]) return nullptr;
    player_params.specs.emplace_back(player_specs[i]);
  }
  scribblez::SelfPlayEngine::Params engine_params;
  engine_params.threads = num_threads;
  engine_params.seed = seed;
  engine_params.handicap_max = handicap_max;
  scribblez::binlog::StreamingGameProducer::Params stream_params;
  stream_params.apply_symmetry = apply_symmetry != 0;
  stream_params.make_encoder = std::move(factory);
  try {
    return new StreamHandle(slot_ptrs, num_slots, rows_per_slot, row_floats, engine_params,
                            player_params, stream_params);
  } catch (const std::exception& e) {
    std::cerr << "new_stream: " << e.what() << "\n";
    return nullptr;
  }
}

}  // namespace

StreamHandle* ScribblezSession::stream_new(float* const* slot_ptrs, int num_slots,
                                           int rows_per_slot, int num_threads, bool post_move,
                                           bool apply_symmetry, uint64_t seed, int handicap_max,
                                           const char* const* player_specs, int num_specs) const {
  const scribblez::InputEncodingSpec enc_spec = spec;
  return ::new_stream(
    slot_ptrs, num_slots, rows_per_slot, num_threads, apply_symmetry, seed, handicap_max,
    player_specs, num_specs, row_size_floats(), [enc_spec, post_move]() {
      return scribblez::binlog::make_position_eval_row_encoder(enc_spec, post_move);
    });
}

StreamHandle* scribblez_stream_new(ScribblezSession* s, float* const* slot_ptrs, int num_slots,
                                   int rows_per_slot, int num_threads, int post_move,
                                   int apply_symmetry, uint64_t seed, int handicap_max,
                                   const char* const* player_specs, int num_specs) {
  return s->stream_new(slot_ptrs, num_slots, rows_per_slot, num_threads, post_move != 0,
                       apply_symmetry != 0, seed, handicap_max, player_specs, num_specs);
}

StreamHandle* ScribblezSession::max_move_per_lane_stream_new(
  float* const* slot_ptrs, int num_slots, int rows_per_slot, int num_threads, bool apply_symmetry,
  uint64_t seed, int handicap_max, const char* const* player_specs, int num_specs) const {
  const scribblez::InputEncodingSpec enc_spec = spec;
  return ::new_stream(
    slot_ptrs, num_slots, rows_per_slot, num_threads, apply_symmetry, seed, handicap_max,
    player_specs, num_specs, scribblez::MaxMovePerLaneTask::kRowFloats,
    [enc_spec]() { return scribblez::binlog::make_max_move_per_lane_row_encoder(enc_spec); });
}

StreamHandle* scribblez_max_move_per_lane_stream_new(ScribblezSession* s, float* const* slot_ptrs,
                                                     int num_slots, int rows_per_slot,
                                                     int num_threads, int apply_symmetry,
                                                     uint64_t seed, int handicap_max,
                                                     const char* const* player_specs,
                                                     int num_specs) {
  return s->max_move_per_lane_stream_new(slot_ptrs, num_slots, rows_per_slot, num_threads,
                                         apply_symmetry != 0, seed, handicap_max, player_specs,
                                         num_specs);
}

void scribblez_stream_start(StreamHandle* h) {
  if (h) h->producer.start();
}

int scribblez_stream_wait_full_slot(StreamHandle* h) {
  if (!h) return -1;
  return h->ring.wait_full_slot();
}

void scribblez_stream_release_slot(StreamHandle* h, int slot) {
  if (h) h->ring.release_slot(slot);
}

void scribblez_stream_get_stats(StreamHandle* h, ScribblezStreamStats* out) {
  if (!h || !out) return;
  const scribblez::binlog::ProducerStats ps = h->producer.stats();
  const scribblez::binlog::RingStats rs = h->ring.stats();
  out->games_played = ps.games_played;
  out->games_dropped = ps.games_dropped;
  out->rows_committed = rs.rows_committed;
  out->slots_published = rs.slots_published;
  out->producer_blocked_ns = rs.producer_blocked_ns;
  out->consumer_blocked_ns = rs.consumer_blocked_ns;
}

void scribblez_stream_stop(StreamHandle* h) {
  if (h) h->producer.stop();
}

void scribblez_stream_delete(StreamHandle* h) { delete h; }

}  // extern "C"
