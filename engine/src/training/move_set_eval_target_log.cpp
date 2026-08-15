#include "training/move_set_eval_target_log.h"

#include "data/binary_log.h"
#include "util/assert.h"
#include "util/exception.h"

#include <Eigen/Core>

#include <array>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <unistd.h>

namespace scribblez {
namespace move_set_eval {

namespace {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

}  // namespace

uint32_t target_flags_from_slog(uint16_t slog_flags) {
  return (slog_flags & binlog::kFlagFaceUpLeaves) != 0 ? kTargetFlagOpenLeaves : 0u;
}

float quantize_plane(const float* values, uint8_t* out) {
  Eigen::Map<const Eigen::ArrayXf> plane(values, kPlaneCells);
  Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>> cells(out, kPlaneCells);
  const float max = plane.maxCoeff();
  if (max <= 0.0f) {
    cells.setZero();
    return 0.0f;
  }
  const float scale = max / 255.0f;
  cells = (plane / scale).round().cast<uint8_t>();
  return scale;
}

TargetWriter::TargetWriter(const std::string& path, uint32_t record_floats, uint32_t record_planes,
                           const std::string& model_hash, uint32_t flags)
    : path_(path), record_floats_(record_floats), record_planes_(record_planes) {
  TargetFileHeader hdr{};
  hdr.magic = kTargetMagic;
  hdr.version = kTargetVersion;
  hdr.num_positions = 0;  // patched in close()
  hdr.record_floats = record_floats;
  hdr.record_planes = record_planes;
  hdr.flags = flags;
  std::strncpy(hdr.model_hash, model_hash.c_str(), sizeof(hdr.model_hash) - 1);
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

TargetWriter::~TargetWriter() {
  if (!closed_) close();
}

void TargetWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                const std::vector<Move>& candidates,
                                const std::vector<float>& targets, const std::vector<float>& planes,
                                uint32_t num_legal_moves) {
  RELEASE_ASSERT(!closed_);
  RELEASE_ASSERT(targets.size() == candidates.size() * record_floats_);
  RELEASE_ASSERT(planes.size() == candidates.size() * record_planes_ * kPlaneCells);
  TargetPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = static_cast<uint32_t>(candidates.size());
  ph.num_legal_moves = num_legal_moves;
  append_bytes(&buffer_, &ph, sizeof(ph));
  std::array<uint8_t, kPlaneCells> quantized;
  for (size_t c = 0; c < candidates.size(); ++c) {
    append_bytes(&buffer_, &candidates[c], sizeof(Move));
    append_bytes(&buffer_, targets.data() + c * record_floats_, sizeof(float) * record_floats_);
    // Scales first, then the quantized planes, so each stays a contiguous
    // fixed-width block (the reader's accessors and the numpy dtype both
    // address them as such).
    const float* cand_planes = planes.data() + c * record_planes_ * kPlaneCells;
    const size_t scales_offset = buffer_.size();
    buffer_.resize(buffer_.size() + sizeof(float) * record_planes_);
    for (uint32_t h = 0; h < record_planes_; ++h) {
      const float scale = quantize_plane(cand_planes + h * kPlaneCells, quantized.data());
      std::memcpy(buffer_.data() + scales_offset + h * sizeof(float), &scale, sizeof(float));
      append_bytes(&buffer_, quantized.data(), kPlaneCells);
    }
  }
  ++num_positions_;
}

void TargetWriter::close() {
  RELEASE_ASSERT(!closed_);
  closed_ = true;
  TargetFileHeader* hdr = reinterpret_cast<TargetFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .mset appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = std::format("{}.tmp.{}", path_, ::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw util::Exception("TargetWriter: cannot open {}", tmp);
    f.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

TargetReader::TargetReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw util::Exception("TargetReader: cannot open {}", path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(static_cast<size_t>(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(TargetFileHeader)) {
    throw util::Exception("TargetReader: truncated header in {}", path);
  }
  std::memcpy(&header_, buffer_.data(), sizeof(header_));
  if (header_.magic != kTargetMagic) {
    throw util::Exception("TargetReader: bad magic in {}", path);
  }
  if (header_.version != kTargetVersion) {
    throw util::Exception("TargetReader: version mismatch in {} (file={} code={})", path,
                          header_.version, kTargetVersion);
  }

  size_t off = sizeof(TargetFileHeader);
  positions_.reserve(header_.num_positions);
  for (uint32_t p = 0; p < header_.num_positions; ++p) {
    if (off + sizeof(TargetPositionHeader) > buffer_.size()) {
      throw util::Exception("TargetReader: truncated position header in {}", path);
    }
    const TargetPositionHeader* ph =
      reinterpret_cast<const TargetPositionHeader*>(buffer_.data() + off);
    off += sizeof(TargetPositionHeader);
    const size_t bytes = static_cast<size_t>(ph->num_candidates) * record_bytes();
    if (off + bytes > buffer_.size()) {
      throw util::Exception("TargetReader: truncated records in {}", path);
    }
    positions_.push_back({ph, buffer_.data() + off});
    off += bytes;
  }
}

std::string TargetReader::model_hash() const {
  return std::string(header_.model_hash, strnlen(header_.model_hash, sizeof(header_.model_hash)));
}

Move TargetReader::move_at(const Position& p, int candidate) const {
  Move m;
  std::memcpy(&m, p.records + candidate * record_bytes(), sizeof(Move));
  return m;
}

const float* TargetReader::targets_at(const Position& p, int candidate) const {
  return reinterpret_cast<const float*>(p.records + candidate * record_bytes() + sizeof(Move));
}

const float* TargetReader::plane_scales_at(const Position& p, int candidate) const {
  return reinterpret_cast<const float*>(p.records + candidate * record_bytes() + sizeof(Move) +
                                        sizeof(float) * header_.record_floats);
}

const uint8_t* TargetReader::planes_at(const Position& p, int candidate) const {
  return reinterpret_cast<const uint8_t*>(plane_scales_at(p, candidate) + header_.record_planes);
}

}  // namespace move_set_eval
}  // namespace scribblez
