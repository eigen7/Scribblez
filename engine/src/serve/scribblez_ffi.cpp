#include "serve/scribblez_ffi.h"

#include "agent/agent.h"
#include "belief/leave_params.h"
#include "belief/leave_sampler.h"
#include "data/binary_log.h"
#include "data/block_decoder.h"
#include "data/data_loader.h"
#include "data/forgone_play_log.h"
#include "data/format_layout.h"
#include "data/gcg_reader.h"
#include "encoding/game_state_encoder.h"
#include "encoding/input_encoder.h"
#include "encoding/position_encoder.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/leave_value_arrays.h"
#include "lexicon/lexicon.h"
#include "sim/sim_runner.h"
#include "sim/win_pct_table.h"
#include "sim/win_table_feature.h"
#include "training/footprint_collapse.h"
#include "training/leaf_eval_task.h"
#include "training/leave_inference_task.h"
#include "training/position_eval_analysis.h"
#include "training/training_targets.h"
#include "util/exception.h"
#include "util/work_pool.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using scribblez::binlog::DataLoader;
using scribblez::binlog::FileHeader;
using scribblez::binlog::kMagic;
using scribblez::binlog::kVersion;

// The session behind the C ABI (scribblez_ffi.h). Owns the InputEncodingSpec,
// the loaded dictionary plus the encoding arm, that every encoding, analysis,
// and loader entry point works from, and the input shapes the shape queries
// report.
struct ScribblezSession {
  ScribblezSession(const char* lexicon_name, bool opp_leave_input);

  const ScribblezShape* input_shapes() const { return input_shapes_; }
  int input_floats() const { return scribblez::input_floats(spec); }

  int decode_rows(int task, const char* path, const int64_t* game_idx, const int64_t* turn_idx,
                  int64_t n, bool post_move, bool hide_private, float* out) const;
  int64_t win_table_samples(const char* path, void* out, int64_t cap) const;
  int position_eval_analyze_gcg(const char* gcg_text, bool opp_leave_input, float* out_input,
                                int input_cap, char* out_err, int err_cap) const;
  int position_eval_analyze_gcg_leaves(const char* gcg_text, const char* leave_str,
                                       const char* opp_leave_str, bool opp_leave_input,
                                       float* out_input, int input_cap, char* out_err,
                                       int err_cap) const;
  int position_eval_collapse_placement(const char* gcg_text, const float* raw, int raw_cap,
                                       float* out, int out_cap, char* out_err, int err_cap) const;
  int position_eval_masked_placement(const char* gcg_text, const float* raw, int raw_cap,
                                     float* out, int out_cap, char* out_err, int err_cap) const;
  int position_eval_legal_placement(const char* gcg_text, float* out, int out_cap, char* out_err,
                                    int err_cap) const;
  DataLoaderHandle* dl_new(int64_t memory_budget, int num_worker_threads, int num_prefetch_threads,
                           int task) const;

  scribblez::InputEncodingSpec spec;

 private:
  std::optional<scribblez::InputEncodingSpec> analysis_arm(bool opp_leave_input, int input_cap,
                                                           char* out_err, int err_cap) const;

  // The dim arrays live here so the advertised pointers stay valid for the
  // session's lifetime.
  int spatial_dims_[3];
  int scalar_dims_[1];
  ScribblezShape input_shapes_[3];
};

namespace {

// Throws, after printing an install hint, when the .kwg is missing.
const scribblez::Dictionary& load_session_dictionary(const char* lexicon_name) {
  scribblez::Lexicon& lex = scribblez::Lexicon::instance();
  lex.set_params({.name = lexicon_name, .dir = lex.dir()});
  return scribblez::load_dictionary_or_throw();
}

}  // namespace

ScribblezSession::ScribblezSession(const char* lexicon_name, bool opp_leave_input)
    : spec{&load_session_dictionary(lexicon_name), opp_leave_input},
      spatial_dims_{scribblez::spatial_planes(), scribblez::kBoardSide, scribblez::kBoardSide},
      scalar_dims_{scribblez::scalar_floats(spec)},
      input_shapes_{{"input_spatial", spatial_dims_, 3, -1},
                    {"input_scalar", scalar_dims_, 1, -1},
                    {nullptr, nullptr, 0, 0}} {}

ScribblezSession* scribblez_session_new(const char* lexicon_name, int opp_leave_input) {
  return new ScribblezSession(lexicon_name, opp_leave_input != 0);
}

void scribblez_session_delete(ScribblezSession* s) { delete s; }

void scribblez_set_worker_count(int n) { scribblez::util::set_worker_count(n); }

namespace {

// The NULL-terminated target shape table, generated at compile time from
// AllTargets so it tracks training_targets.h with no edits here.
template <typename List>
struct TargetShapeTable;

template <typename... Ts>
struct TargetShapeTable<scribblez::TargetList<Ts...>> {
  static constexpr std::size_t kCount = sizeof...(Ts);
  static constexpr std::array<ScribblezShape, kCount + 1> kValue = []() {
    std::array<ScribblezShape, kCount + 1> a{};
    std::size_t i = 0;
    (void)std::initializer_list<int>{
      (a[i] = ScribblezShape{Ts::kName, Ts::kDims, int(std::size(Ts::kDims)), int(i)}, ++i, 0)...};
    a[kCount] = ScribblezShape{nullptr, nullptr, 0, 0};
    return a;
  }();
};

constexpr auto kTargetShapesArr = TargetShapeTable<scribblez::AllTargets>::kValue;

// The leave inference task's shapes (training/leave_inference_task.h).
constexpr int kLeaveInferenceSpatialDims[3] = {
  scribblez::LeaveInferenceInputEncoder::kSpatialPlanes, scribblez::kBoardSide,
  scribblez::kBoardSide};
constexpr int kLeaveInferenceScalarDims[1] = {scribblez::LeaveInferenceInputEncoder::kScalarFloats};

const ScribblezShape kLeaveInferenceInputShapes[] = {
  {"input_spatial", kLeaveInferenceSpatialDims, 3, -1},
  {"input_scalar", kLeaveInferenceScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

constexpr int kOppLeaveTilesDims[1] = {scribblez::OppLeaveTarget::kTiles};
constexpr int kScalarTargetDims[1] = {1};
constexpr int kPoolTargetDims[1] = {scribblez::TILE_KINDS};
static_assert(scribblez::OppLeaveTarget::kFloats ==
                scribblez::OppLeaveTarget::kTiles + 4 + scribblez::TILE_KINDS,
              "the target shapes below must cover OppLeaveTarget's floats");

const ScribblezShape kLeaveInferenceTargetShapes[] = {
  {"opp_leave_tiles", kOppLeaveTilesDims, 1, 0},
  {"opp_leave_length", kScalarTargetDims, 1, 1},
  {"opp_leave_valid", kScalarTargetDims, 1, 2},
  {"opp_leave_cause", kScalarTargetDims, 1, 3},
  {"leave_pool", kPoolTargetDims, 1, 4},
  {"bag_size", kScalarTargetDims, 1, 5},
  {nullptr, nullptr, 0, 0},
};

// The leaf evaluation task's shapes (training/leaf_eval_task.h).
constexpr int kLeafEvalSpatialDims[3] = {scribblez::LeafEvalInputEncoder::kSpatialPlanes,
                                         scribblez::kBoardSide, scribblez::kBoardSide};
constexpr int kLeafEvalScalarDims[1] = {scribblez::LeafEvalInputEncoder::kScalarFloats};

const ScribblezShape kLeafEvalInputShapes[] = {
  {"input_spatial", kLeafEvalSpatialDims, 3, -1},
  {"input_scalar", kLeafEvalScalarDims, 1, -1},
  {nullptr, nullptr, 0, 0},
};

const ScribblezShape kLeafEvalTargetShapes[] = {
  {scribblez::WldTarget::kName, scribblez::WldTarget::kDims, 1, 0},
  {scribblez::ScoreDiffTarget::kName, scribblez::ScoreDiffTarget::kDims, 1, 1},
  {scribblez::OwnershipTarget::kName, scribblez::OwnershipTarget::kDims, 2, 2},
  {scribblez::SquarePointsTarget::kName, scribblez::SquarePointsTarget::kDims, 2, 3},
  {scribblez::GoingOutTarget::kName, scribblez::GoingOutTarget::kDims, 1, 4},
  {scribblez::EndPointsTarget::kName, scribblez::EndPointsTarget::kDims, 1, 5},
  {nullptr, nullptr, 0, 0},
};

// The DecodeTask a C ABI task code names, if any.
std::optional<scribblez::binlog::DecodeTask> decode_task(int code) {
  using scribblez::binlog::DecodeTask;
  switch (code) {
    case 0:
      return DecodeTask::kPositionEval;
    case 1:
      return DecodeTask::kLeaveInference;
    case 2:
      return DecodeTask::kPublicLeaveInference;
    case 3:
      return DecodeTask::kLeafEval;
  }
  return std::nullopt;
}

// Loads what a task's rows read beyond the session: HastyBot's leave values for
// the leaf eval task. Throws if they cannot load.
void prepare_task(scribblez::binlog::DecodeTask task) {
  if (task == scribblez::binlog::DecodeTask::kLeafEval) {
    scribblez::HastyEquity::ensure_initialized(scribblez::Lexicon::instance().name());
  }
}

// A task's input and target tensors.
struct TaskShapes {
  const ScribblezShape* input;
  const ScribblezShape* target;
};

// The tensors of the task a C ABI task code names, or nulls for an unknown
// code. Position evaluation's inputs follow the session's arm.
TaskShapes task_shapes(const ScribblezSession* s, int code) {
  using scribblez::binlog::DecodeTask;
  const std::optional<DecodeTask> t = decode_task(code);
  if (!t) return {nullptr, nullptr};
  switch (*t) {
    case DecodeTask::kPositionEval:
      return {s->input_shapes(), kTargetShapesArr.data()};
    case DecodeTask::kLeaveInference:
    case DecodeTask::kPublicLeaveInference:
      return {kLeaveInferenceInputShapes, kLeaveInferenceTargetShapes};
    case DecodeTask::kLeafEval:
      return {kLeafEvalInputShapes, kLeafEvalTargetShapes};
  }
  return {nullptr, nullptr};
}

}  // namespace

extern "C" {

const ScribblezShape* scribblez_task_input_shapes(ScribblezSession* s, int task) {
  return task_shapes(s, task).input;
}

const ScribblezShape* scribblez_task_target_shapes(ScribblezSession* s, int task) {
  return task_shapes(s, task).target;
}

int scribblez_task_row_floats(ScribblezSession* s, int task) {
  const std::optional<scribblez::binlog::DecodeTask> t = decode_task(task);
  return t ? scribblez::binlog::BlockDecoder::row_floats_for(*t, s->spec) : -1;
}

namespace {

// Read a .slog file into `buf`, validate its header, and bounds-check
// `game_idx`. Returns 0 on success, -1 on any failure.
int load_slog(const char* path, int64_t game_idx, std::vector<char>& buf) {
  if (!path) return -1;
  std::ifstream f(path, std::ios::binary);
  if (!f) return -1;
  buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < sizeof(FileHeader)) return -1;
  const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
  if (hdr->magic != kMagic || hdr->version != kVersion) return -1;
  if (game_idx >= int64_t(hdr->num_games)) return -1;
  return 0;
}

// Copy `s` into the caller's buffer, NUL-terminated and truncated to out_cap,
// and return its full length so the caller can detect truncation and retry.
int emit_string(const std::string& s, char* out, int out_cap) {
  const int len = s.size();
  if (out && out_cap > 0) {
    const int n = len < out_cap - 1 ? len : out_cap - 1;
    std::memcpy(out, s.data(), size_t(n));
    out[n] = '\0';
  }
  return len;
}

// `n` Moves starting `first` moves into a caller's buffer of serialized
// Moves, copied out because the buffer is not guaranteed to be aligned for
// Move.
std::vector<scribblez::Move> copy_moves(const void* moves, int64_t first, int64_t n) {
  std::vector<scribblez::Move> out(static_cast<size_t>(n));
  std::memcpy(out.data(), static_cast<const char*>(moves) + first * sizeof(scribblez::Move),
              out.size() * sizeof(scribblez::Move));
  return out;
}

// copy_moves, each move transposed (Move::transpose) when `transpose`.
std::vector<scribblez::Move> copy_moves_in_frame(const void* moves, int64_t first, int64_t n,
                                                 bool transpose) {
  std::vector<scribblez::Move> out = copy_moves(moves, first, n);
  if (transpose)
    for (scribblez::Move& m : out) m = m.transpose();
  return out;
}

// Copies `count` elements of `size` bytes from `data` to `out` if `count` is
// `expected`. Not a template: this file's helpers sit in the C ABI's linkage.
bool emit_exact(const void* data, size_t count, size_t size, int64_t expected, void* out) {
  if (int64_t(count) != expected) return false;
  std::memcpy(out, data, count * size);
  return true;
}

}  // namespace

int ScribblezSession::decode_rows(int task, const char* path, const int64_t* game_idx,
                                  const int64_t* turn_idx, int64_t n, bool post_move,
                                  bool hide_private, float* out) const {
  const std::optional<scribblez::binlog::DecodeTask> t = decode_task(task);
  if (!t || !game_idx || !turn_idx || !out || n < 0) return -1;
  std::vector<char> buf;
  if (load_slog(path, /*game_idx=*/0, buf) != 0) return -1;
  try {
    prepare_task(*t);
    std::optional<scribblez::ForgoneSidecar> forgone;
    if (scribblez::binlog::BlockDecoder::reads_forgone(*t)) {
      forgone.emplace(scribblez::forgone_path_of(path), buf.data(), spec.dict->content_hash());
    }
    scribblez::binlog::BlockDecoder decoder(spec, *t);
    const int64_t row_floats = scribblez::binlog::BlockDecoder::row_floats_for(*t, spec);
    for (int64_t j = 0; j < n; ++j) {
      decoder.decode_one(buf.data(), path, uint32_t(game_idx[j]), uint32_t(turn_idx[j]),
                         {.hide_private = hide_private}, post_move,
                         /*output_row=*/0, out + j * row_floats, forgone ? &*forgone : nullptr);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "scribblez_decode_rows: " << e.what() << "\n";
    return -1;
  }
}

int scribblez_decode_rows(ScribblezSession* s, int task, const char* path, const int64_t* game_idx,
                          const int64_t* turn_idx, int64_t n, int post_move, int hide_private,
                          float* out) {
  return s->decode_rows(task, path, game_idx, turn_idx, n, post_move != 0, hide_private != 0, out);
}

int64_t ScribblezSession::win_table_samples(const char* path, void* out, int64_t cap) const {
  std::vector<char> buf;
  if (load_slog(path, /*game_idx=*/0, buf) != 0) return -1;
  // Win tables model hidden-leaves play only.
  if (reinterpret_cast<const FileHeader*>(buf.data())->flags & scribblez::binlog::kFlagFaceUpLeaves)
    return -2;
  scribblez::HastyEquity::ensure_initialized(scribblez::Lexicon::instance().name());
  std::vector<scribblez::WinTableSample> samples;
  scribblez::append_win_table_samples(buf.data(), samples);
  const int64_t n = std::min(int64_t(samples.size()), cap);
  if (n > 0) std::memcpy(out, samples.data(), size_t(n) * sizeof(scribblez::WinTableSample));
  return int64_t(samples.size());
}

int64_t scribblez_win_table_samples(ScribblezSession* s, const char* path, void* out_samples,
                                    int64_t cap) {
  return s->win_table_samples(path, out_samples, cap);
}

namespace {

// "key=value" lines into a map; a line without '=' is an error.
std::map<std::string, std::string> parse_header_lines(const char* text) {
  std::map<std::string, std::string> out;
  std::istringstream in(text ? text : "");
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const size_t eq = line.find('=');
    if (eq == std::string::npos) throw scribblez::util::Exception("bad header line '{}'", line);
    out[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return out;
}

}  // namespace

struct LeaveValueArraysHandle {
  scribblez::LeaveValueArrays arrays;
};

LeaveValueArraysHandle* scribblez_leave_value_arrays_new(ScribblezSession* /*s*/) {
  // The session has set the lexicon whose leave values these are.
  scribblez::HastyEquity::ensure_initialized(scribblez::Lexicon::instance().name());
  return new LeaveValueArraysHandle{
    scribblez::LeaveValueArrays(scribblez::HastyEquity::instance().leave_values())};
}

void scribblez_leave_value_arrays_delete(LeaveValueArraysHandle* a) { delete a; }

int64_t scribblez_leave_value_arrays_rows(const LeaveValueArraysHandle* a, int k) {
  if (k < 1 || k > scribblez::kLeaveArraysMaxLength) return -1;
  return int64_t(a->arrays.rows(k));
}

int scribblez_leave_value_arrays_copy(const LeaveValueArraysHandle* a, int k, uint8_t* out_codes,
                                      float* out_values) {
  if (k < 1 || k > scribblez::kLeaveArraysMaxLength) return -1;
  const size_t n = a->arrays.rows(k);
  for (int slot = 0; slot < k; ++slot) {
    std::memcpy(out_codes + size_t(slot) * n, a->arrays.codes(k, slot).data(), n);
  }
  std::memcpy(out_values, a->arrays.values(k).data(), n * sizeof(float));
  return 0;
}

const char* scribblez_leave_value_arrays_content_hash(const LeaveValueArraysHandle* a) {
  return a->arrays.content_hash().c_str();
}

namespace {

scribblez::TileCounts tile_counts(const int32_t* counts) {
  scribblez::TileCounts out;
  for (int t = 0; t < scribblez::TILE_KINDS; ++t) out.add(scribblez::Tile::of(t), counts[t]);
  return out;
}

scribblez::belief::LeaveSampler make_sampler(const LeaveValueArraysHandle* a, const float* params,
                                             int k, const int32_t* leave_pool) {
  scribblez::belief::LeaveParams p;
  std::memcpy(&p, params, sizeof(p));
  return scribblez::belief::LeaveSampler(p, k, tile_counts(leave_pool), a->arrays);
}

}  // namespace

int scribblez_leave_sampler_query(const LeaveValueArraysHandle* a, const float* params, int k,
                                  const int32_t* leave_pool, const int64_t* rows, int64_t n_rows,
                                  const double* quantiles, int64_t n_quantiles, double* out_log_z,
                                  double* out_log_probs, int64_t* out_samples, char* out_err,
                                  int err_cap) {
  try {
    const scribblez::belief::LeaveSampler sampler = make_sampler(a, params, k, leave_pool);
    *out_log_z = sampler.log_z();
    for (int64_t i = 0; i < n_rows; ++i) out_log_probs[i] = sampler.log_prob(size_t(rows[i]));
    for (int64_t i = 0; i < n_quantiles; ++i)
      out_samples[i] = int64_t(sampler.sample(quantiles[i]));
    return 0;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_win_table_write(const char* path, const float* grid, const char* header,
                              char* out_err, int err_cap) {
  try {
    scribblez::WinPctTable::from_grid(grid, parse_header_lines(header)).write_versioned(path);
    return 0;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_win_table_eval(const char* path, int macondo_format, const void* features, int64_t n,
                             float* out, char* out_err, int err_cap) {
  try {
    const scribblez::WinPctTable table = macondo_format
                                           ? scribblez::WinPctTable::load_macondo(path)
                                           : scribblez::WinPctTable::load_versioned(path);
    const char* bytes = static_cast<const char*>(features);
    for (int64_t i = 0; i < n; ++i) {
      scribblez::WinTableFeature f;
      std::memcpy(&f, bytes + i * sizeof(f), sizeof(f));
      out[i] = table.win_prob(f);
    }
    return 0;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

// The encoding spec for an analysis entry point's explicit arm. Returns
// nullopt, with the mismatch in `out_err`, when the caller's buffer (sized for
// its model's input) does not match the arm's width: that model cannot be fed
// the engine's rows.
std::optional<scribblez::InputEncodingSpec> ScribblezSession::analysis_arm(bool opp_leave_input,
                                                                           int input_cap,
                                                                           char* out_err,
                                                                           int err_cap) const {
  const scribblez::InputEncodingSpec arm{spec.dict, opp_leave_input};
  if (scribblez::input_floats(arm) != input_cap) {
    emit_string(std::format("input buffer holds {} floats but the arm encodes {}", input_cap,
                            scribblez::input_floats(arm)),
                out_err, err_cap);
    return std::nullopt;
  }
  return arm;
}

int ScribblezSession::position_eval_analyze_gcg(const char* gcg_text, bool opp_leave_input,
                                                float* out_input, int input_cap, char* out_err,
                                                int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !out_input) return -1;
  try {
    const auto arm = analysis_arm(opp_leave_input, input_cap, out_err, err_cap);
    if (!arm) return -1;
    std::string error;
    if (!scribblez::encode_position_eval_analysis_input(gcg_text, *arm, out_input, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return input_cap;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_position_eval_analyze_gcg(ScribblezSession* s, const char* gcg_text,
                                        int opp_leave_input, float* out_input, int input_cap,
                                        char* out_err, int err_cap) {
  return s->position_eval_analyze_gcg(gcg_text, opp_leave_input != 0, out_input, input_cap, out_err,
                                      err_cap);
}

int ScribblezSession::position_eval_collapse_placement(const char* gcg_text, const float* raw,
                                                       int raw_cap, float* out, int out_cap,
                                                       char* out_err, int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !raw || !out) return -1;
  constexpr int kRaw = scribblez::kPlacementHeads * scribblez::kFootprintClasses;
  constexpr int kOut =
    scribblez::kPlacementHeads * scribblez::kFootprintSide * scribblez::kFootprintSide;
  if (raw_cap < kRaw || out_cap < kOut) {
    emit_string("placement collapse buffer too small", out_err, err_cap);
    return -1;
  }
  try {
    std::string error;
    if (!scribblez::collapse_position_eval_analysis_placement(gcg_text, spec, raw, out, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return kOut;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int ScribblezSession::position_eval_masked_placement(const char* gcg_text, const float* raw,
                                                     int raw_cap, float* out, int out_cap,
                                                     char* out_err, int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !raw || !out) return -1;
  constexpr int kRaw = scribblez::kPlacementHeads * scribblez::kFootprintClasses;
  constexpr int kOut = scribblez::kPlacementHeads * scribblez::kFootprintClasses;
  if (raw_cap < kRaw || out_cap < kOut) {
    emit_string("masked placement buffer too small", out_err, err_cap);
    return -1;
  }
  try {
    std::string error;
    if (!scribblez::masked_position_eval_analysis_placement(gcg_text, spec, raw, out, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return kOut;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int ScribblezSession::position_eval_legal_placement(const char* gcg_text, float* out, int out_cap,
                                                    char* out_err, int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !out) return -1;
  constexpr int kOut =
    scribblez::kPlacementHeads * scribblez::kFootprintSide * scribblez::kFootprintSide;
  if (out_cap < kOut) {
    emit_string("legal placement buffer too small", out_err, err_cap);
    return -1;
  }
  try {
    std::string error;
    if (!scribblez::legal_position_eval_analysis_placement(gcg_text, spec, out, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return kOut;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_position_eval_collapse_placement(ScribblezSession* s, const char* gcg_text,
                                               const float* raw, int raw_cap, float* out,
                                               int out_cap, char* out_err, int err_cap) {
  return s->position_eval_collapse_placement(gcg_text, raw, raw_cap, out, out_cap, out_err,
                                             err_cap);
}

int scribblez_position_eval_masked_placement(ScribblezSession* s, const char* gcg_text,
                                             const float* raw, int raw_cap, float* out, int out_cap,
                                             char* out_err, int err_cap) {
  return s->position_eval_masked_placement(gcg_text, raw, raw_cap, out, out_cap, out_err, err_cap);
}

int scribblez_position_eval_legal_placement(ScribblezSession* s, const char* gcg_text, float* out,
                                            int out_cap, char* out_err, int err_cap) {
  return s->position_eval_legal_placement(gcg_text, out, out_cap, out_err, err_cap);
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

int ScribblezSession::position_eval_analyze_gcg_leaves(const char* gcg_text, const char* leave_str,
                                                       const char* opp_leave_str,
                                                       bool opp_leave_input, float* out_input,
                                                       int input_cap, char* out_err,
                                                       int err_cap) const {
  if (out_err && err_cap > 0) out_err[0] = '\0';
  if (!gcg_text || !leave_str || !out_input) return -1;
  try {
    const auto arm = analysis_arm(opp_leave_input, input_cap, out_err, err_cap);
    if (!arm) return -1;
    std::string error;
    const std::optional<std::string> opp_leave =
      opp_leave_str ? std::optional<std::string>(opp_leave_str) : std::nullopt;
    if (!scribblez::encode_position_eval_analysis_input_with_leaves(
          gcg_text, leave_str, opp_leave ? &*opp_leave : nullptr, *arm, out_input, &error)) {
      emit_string(error, out_err, err_cap);
      return -1;
    }
    return input_cap;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int scribblez_position_eval_analyze_gcg_leaves(ScribblezSession* s, const char* gcg_text,
                                               const char* leave_str, const char* opp_leave_str,
                                               int opp_leave_input, float* out_input, int input_cap,
                                               char* out_err, int err_cap) {
  return s->position_eval_analyze_gcg_leaves(gcg_text, leave_str, opp_leave_str,
                                             opp_leave_input != 0, out_input, input_cap, out_err,
                                             err_cap);
}

const char* scribblez_format_layout_json(void) { return scribblez::format_layout_json().c_str(); }

int scribblez_read_file_header(const char* path, int64_t* out_num_games, int64_t* out_file_size) {
  if (!path || !out_num_games || !out_file_size) return -1;
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
  *out_num_games = hdr.num_games;
  *out_file_size = fsz;
  return 0;
}

struct DataLoaderHandle {
  DataLoader loader;
  explicit DataLoaderHandle(const DataLoader::Params& p) : loader(p) {}
};

DataLoaderHandle* ScribblezSession::dl_new(int64_t memory_budget, int num_worker_threads,
                                           int num_prefetch_threads, int task) const {
  const std::optional<scribblez::binlog::DecodeTask> t = decode_task(task);
  if (!t) return nullptr;
  try {
    prepare_task(*t);
  } catch (const std::exception& e) {
    std::cerr << "scribblez_dl_new: " << e.what() << "\n";
    return nullptr;
  }
  DataLoader::Params p;
  p.spec = spec;
  p.task = *t;
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

int scribblez_dl_add_file(DataLoaderHandle* h, const char* path, int64_t num_games,
                          int64_t file_size, char* out_err, int err_cap) {
  if (!h || !path) return -1;
  try {
    h->loader.add_file(path, num_games, file_size);
    return 0;
  } catch (const std::exception& e) {
    emit_string(e.what(), out_err, err_cap);
    return -1;
  }
}

int64_t scribblez_dl_num_positions(const DataLoaderHandle* h) {
  if (!h) return 0;
  return h->loader.num_positions();
}

int scribblez_dl_epoch_start(DataLoaderHandle* h, int batch_size, int post_move, int apply_symmetry,
                             uint64_t seed, int turns_per_game, int epoch_index,
                             double hide_private_fraction) {
  if (!h) return 0;
  DataLoader::EpochConfig cfg;
  cfg.batch_size = batch_size;
  cfg.post_move = post_move != 0;
  cfg.apply_symmetry = apply_symmetry != 0;
  cfg.seed = seed;
  cfg.turns_per_game = turns_per_game;
  cfg.epoch_index = epoch_index;
  cfg.hide_private_fraction = hide_private_fraction;
  return h->loader.epoch_start(cfg);
}

int scribblez_dl_load_batch(DataLoaderHandle* h, float* output) {
  if (!h || !output) return 0;
  // load_batch throws when a file becomes unreadable mid-epoch; the Python
  // wrapper raises on -1.
  try {
    return h->loader.load_batch(output);
  } catch (const std::exception& e) {
    std::cerr << "scribblez_dl_load_batch: " << e.what() << "\n";
    return -1;
  }
}

}  // extern "C"
