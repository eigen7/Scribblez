#include "data/format_layout.h"

#include "data/binary_log.h"
#include "data/sim_observation_log.h"
#include "game/glyph.h"
#include "game/move.h"
#include "game/rack.h"
#include "sim/sim_runner.h"
#include "training/move_set_eval_target_log.h"
#include "training/training_targets.h"

#include <boost/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace scribblez {

namespace {

namespace json = boost::json;

// How each member type describes itself into a field object -- one mapping
// per line. Scalars carry a numpy dtype code (little-endian, matching the
// on-disk packed structs); one-byte enums and wrappers serialize as their
// underlying byte; std::array members add a shape; char arrays are numpy
// fixed bytes; structs that are themselves described reference their entry
// by name; Rack is opaque to readers and maps to a numpy void of its size.
template <typename T>
struct MemberDtype;

#define SCRIBBLEZ_DTYPE(T, code)                                    \
  template <>                                                       \
  struct MemberDtype<T> {                                           \
    static void describe(json::object* f) { (*f)["dtype"] = code; } \
  }
#define SCRIBBLEZ_STRUCT_REF(T)                                                             \
  template <>                                                                               \
  struct MemberDtype<T> {                                                                   \
    static void describe(json::object* f) { (*f)["dtype"] = json::object{{"struct", #T}}; } \
  }

SCRIBBLEZ_DTYPE(uint8_t, "u1");
SCRIBBLEZ_DTYPE(int8_t, "i1");
SCRIBBLEZ_DTYPE(uint16_t, "<u2");
SCRIBBLEZ_DTYPE(int16_t, "<i2");
SCRIBBLEZ_DTYPE(uint32_t, "<u4");
SCRIBBLEZ_DTYPE(int32_t, "<i4");
SCRIBBLEZ_DTYPE(uint64_t, "<u8");
SCRIBBLEZ_DTYPE(int64_t, "<i8");
SCRIBBLEZ_DTYPE(float, "<f4");
SCRIBBLEZ_DTYPE(bool, "u1");
SCRIBBLEZ_DTYPE(MoveType, "u1");
SCRIBBLEZ_DTYPE(Glyph, "u1");
SCRIBBLEZ_DTYPE(Rack, "V" + std::to_string(sizeof(Rack)));
SCRIBBLEZ_STRUCT_REF(Move);
SCRIBBLEZ_STRUCT_REF(SimObservation);

#undef SCRIBBLEZ_DTYPE
#undef SCRIBBLEZ_STRUCT_REF

template <std::size_t N>
struct MemberDtype<char[N]> {
  static void describe(json::object* f) { (*f)["dtype"] = "S" + std::to_string(N); }
};

template <typename E, std::size_t N>
struct MemberDtype<std::array<E, N>> {
  static void describe(json::object* f) {
    MemberDtype<E>::describe(f);
    (*f)["shape"] = json::array{N};
  }
};

// Accumulates one struct's fields and enforces coverage: the described
// fields must tile sizeof(struct) exactly. These formats are packed, with
// any reserved byte an explicit member, so every byte belongs to exactly one
// member -- and a member added to the struct without being described here
// leaves a hole that build() rejects. That turns the field lists' apparent
// duplication into a checked restatement: add-a-field breaks the coverage
// sum, remove/rename breaks offsetof at compile time, and a type or order
// change re-derives automatically.
class StructBuilder {
 public:
  StructBuilder(const char* name, std::size_t itemsize) : name_(name), itemsize_(itemsize) {}

  template <typename T>
  void add(const char* field_name, std::size_t offset) {
    json::object f;
    f["name"] = field_name;
    f["offset"] = offset;
    MemberDtype<T>::describe(&f);
    fields_.push_back(std::move(f));
    covered_ += sizeof(T);
  }

  json::object build() && {
    if (covered_ != itemsize_) {
      throw std::logic_error(std::string("format_layout: fields of ") + name_ + " cover " +
                             std::to_string(covered_) + " of " + std::to_string(itemsize_) +
                             " bytes; a member was added without describing it");
    }
    return {{"itemsize", itemsize_}, {"fields", std::move(fields_)}};
  }

 private:
  const char* name_;
  std::size_t itemsize_;
  std::size_t covered_ = 0;
  json::array fields_;
};

// The stringized member is the reader-facing field name, so it must be the
// clean on-disk name -- use FIELD only on structs whose members carry no
// trailing underscore; FIELD_AS names the field explicitly.
#define FIELD(b, Struct, member) \
  (b).add<decltype(Struct::member)>(#member, offsetof(Struct, member))
#define FIELD_AS(b, Struct, member, name) \
  (b).add<decltype(Struct::member)>(name, offsetof(Struct, member))

json::object build_constants() {
  json::object c;
  c["board_size"] = BOARD_SIZE;
  c["slog"] = {{"magic", binlog::kMagic},
               {"version", binlog::kVersion},
               {"flag_face_up_leaves", binlog::kFlagFaceUpLeaves}};
  c["sobs"] = {{"magic", kSimObsMagic},
               {"version", kSimObsVersion},
               {"flag_retired_open_rack", 1},
               {"flag_open_leaves", kSimObsFlagOpenLeaves}};
  {
    json::array target_names;
    for (const char* name : move_set_eval::kTargetNamesV1) target_names.emplace_back(name);
    c["mset"] = {{"magic", move_set_eval::kTargetMagic},
                 {"version", move_set_eval::kTargetVersion},
                 {"flag_open_leaves", move_set_eval::kTargetFlagOpenLeaves},
                 {"flag_full_sweep", move_set_eval::kTargetFlagFullSweep},
                 {"target_names_v1", std::move(target_names)},
                 {"planes", move_set_eval::kTargetPlanes},
                 {"plane_cells", move_set_eval::kPlaneCells}};
  }
  c["move_type"] = {{"play", static_cast<int>(MoveType::PLAY)},
                    {"exchange", static_cast<int>(MoveType::EXCHANGE)},
                    {"pass", static_cast<int>(MoveType::PASS)}};
  // The four placement heads in declaration order (training_targets.h) --
  // also the .mset plane order and the SimObservation count-plane order.
  c["placement_head_names"] = {OppNextPlacementTarget::kName, SelfNextPlacementTarget::kName,
                               OppWinPlacementTarget::kName, SelfWinPlacementTarget::kName};
  return c;
}

json::object build_structs();  // needs FormatFieldAccess, defined after it

json::value build() { return {{"structs", build_structs()}, {"constants", build_constants()}}; }

}  // namespace

// Move's members are private; this friend (declared in move.h) is the one
// place outside the class that reads their offsets.
struct FormatFieldAccess {
  static json::object move_struct() {
    StructBuilder b("Move", sizeof(Move));
    FIELD_AS(b, Move, type_, "type");
    FIELD_AS(b, Move, horizontal_, "horizontal");
    FIELD_AS(b, Move, start_, "start");
    FIELD_AS(b, Move, num_played_, "num_played");
    FIELD_AS(b, Move, glyphs_, "glyphs");
    FIELD_AS(b, Move, reserved_, "reserved");
    FIELD_AS(b, Move, square_mask_, "square_mask");
    FIELD_AS(b, Move, score_, "score");
    return std::move(b).build();
  }
};

namespace {

json::object build_structs() {
  using binlog::FileHeader;
  using binlog::GameMetadata;
  using binlog::InitialRacks;
  using binlog::TurnBlob;
  using move_set_eval::TargetFileHeader;
  using move_set_eval::TargetPositionHeader;

  json::object s;
  s["Move"] = FormatFieldAccess::move_struct();
  {
    StructBuilder b("SimObservation", sizeof(SimObservation));
    FIELD(b, SimObservation, n);
    FIELD(b, SimObservation, wins);
    FIELD(b, SimObservation, draws);
    FIELD(b, SimObservation, losses);
    FIELD(b, SimObservation, delta_sum);
    FIELD(b, SimObservation, delta_sq_sum);
    FIELD(b, SimObservation, opp_next_count);
    FIELD(b, SimObservation, self_next_count);
    FIELD(b, SimObservation, opp_win_count);
    FIELD(b, SimObservation, self_win_count);
    s["SimObservation"] = std::move(b).build();
  }
  {
    StructBuilder b("SobsFileHeader", sizeof(SimObsFileHeader));
    FIELD(b, SimObsFileHeader, magic);
    FIELD(b, SimObsFileHeader, version);
    FIELD(b, SimObsFileHeader, reserved);
    FIELD(b, SimObsFileHeader, num_positions);
    FIELD(b, SimObsFileHeader, flags);
    s["SobsFileHeader"] = std::move(b).build();
  }
  {
    StructBuilder b("SobsPositionHeader", sizeof(SimObsPositionHeader));
    FIELD(b, SimObsPositionHeader, game_index);
    FIELD(b, SimObsPositionHeader, turn_index);
    FIELD(b, SimObsPositionHeader, num_candidates);
    FIELD(b, SimObsPositionHeader, rollouts);
    FIELD(b, SimObsPositionHeader, base_seed);
    s["SobsPositionHeader"] = std::move(b).build();
  }
  {
    StructBuilder b("SobsRecord", sizeof(SimObsRecord));
    FIELD(b, SimObsRecord, move);
    FIELD(b, SimObsRecord, obs);
    s["SobsRecord"] = std::move(b).build();
  }
  {
    StructBuilder b("MsetFileHeader", sizeof(TargetFileHeader));
    FIELD(b, TargetFileHeader, magic);
    FIELD(b, TargetFileHeader, version);
    FIELD(b, TargetFileHeader, reserved);
    FIELD(b, TargetFileHeader, num_positions);
    FIELD(b, TargetFileHeader, record_floats);
    FIELD(b, TargetFileHeader, record_planes);
    FIELD(b, TargetFileHeader, flags);
    FIELD(b, TargetFileHeader, model_hash);
    s["MsetFileHeader"] = std::move(b).build();
  }
  {
    StructBuilder b("MsetPositionHeader", sizeof(TargetPositionHeader));
    FIELD(b, TargetPositionHeader, game_index);
    FIELD(b, TargetPositionHeader, turn_index);
    FIELD(b, TargetPositionHeader, num_candidates);
    FIELD(b, TargetPositionHeader, num_legal_moves);
    s["MsetPositionHeader"] = std::move(b).build();
  }
  {
    StructBuilder b("SlogFileHeader", sizeof(FileHeader));
    FIELD(b, FileHeader, magic);
    FIELD(b, FileHeader, version);
    FIELD(b, FileHeader, flags);
    FIELD(b, FileHeader, num_games);
    FIELD(b, FileHeader, num_sample_positions);
    s["SlogFileHeader"] = std::move(b).build();
  }
  {
    StructBuilder b("SlogGameMetadata", sizeof(GameMetadata));
    FIELD(b, GameMetadata, start_offset);
    FIELD(b, GameMetadata, num_turns);
    FIELD(b, GameMetadata, sampled_turn);
    FIELD(b, GameMetadata, final_score_p0);
    FIELD(b, GameMetadata, final_score_p1);
    FIELD(b, GameMetadata, initial_score_p0);
    FIELD(b, GameMetadata, initial_score_p1);
    FIELD(b, GameMetadata, eligible_begin);
    FIELD(b, GameMetadata, eligible_end);
    s["SlogGameMetadata"] = std::move(b).build();
  }
  {
    StructBuilder b("SlogInitialRacks", sizeof(InitialRacks));
    FIELD(b, InitialRacks, p0);
    FIELD(b, InitialRacks, p1);
    s["SlogInitialRacks"] = std::move(b).build();
  }
  {
    StructBuilder b("SlogTurnBlob", sizeof(TurnBlob));
    FIELD(b, TurnBlob, move);
    FIELD(b, TurnBlob, drawn);
    s["SlogTurnBlob"] = std::move(b).build();
  }
  return s;
}

}  // namespace

const std::string& format_layout_json() {
  static const std::string doc = json::serialize(build());
  return doc;
}

}  // namespace scribblez
