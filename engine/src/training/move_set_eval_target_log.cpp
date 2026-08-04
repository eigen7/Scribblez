#include "training/move_set_eval_target_log.h"

#include "data/binary_log.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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

TargetWriter::TargetWriter(const std::string& path, uint32_t record_floats,
                           const std::string& model_hash, uint32_t flags)
    : path_(path), record_floats_(record_floats) {
  TargetFileHeader hdr{};
  hdr.magic = kTargetMagic;
  hdr.version = kTargetVersion;
  hdr.num_positions = 0;  // patched in close()
  hdr.record_floats = record_floats;
  hdr.flags = flags;
  std::strncpy(hdr.model_hash, model_hash.c_str(), sizeof(hdr.model_hash) - 1);
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

TargetWriter::~TargetWriter() {
  if (!closed_) close();
}

void TargetWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                const std::vector<Move>& candidates,
                                const std::vector<float>& targets, uint32_t num_legal_moves) {
  assert(!closed_);
  assert(targets.size() == candidates.size() * record_floats_);
  TargetPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = static_cast<uint32_t>(candidates.size());
  ph.num_legal_moves = num_legal_moves;
  append_bytes(&buffer_, &ph, sizeof(ph));
  for (size_t c = 0; c < candidates.size(); ++c) {
    append_bytes(&buffer_, &candidates[c], sizeof(Move));
    append_bytes(&buffer_, targets.data() + c * record_floats_, sizeof(float) * record_floats_);
  }
  ++num_positions_;
}

void TargetWriter::close() {
  assert(!closed_);
  closed_ = true;
  TargetFileHeader* hdr = reinterpret_cast<TargetFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .mset appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = path_ + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("TargetWriter: cannot open " + tmp);
    f.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

TargetReader::TargetReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("TargetReader: cannot open " + path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(static_cast<size_t>(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(TargetFileHeader)) {
    throw std::runtime_error("TargetReader: truncated header in " + path);
  }
  std::memcpy(&header_, buffer_.data(), sizeof(header_));
  if (header_.magic != kTargetMagic) {
    throw std::runtime_error("TargetReader: bad magic in " + path);
  }
  if (header_.version != kTargetVersion) {
    throw std::runtime_error("TargetReader: version mismatch in " + path +
                             " (file=" + std::to_string(header_.version) +
                             " code=" + std::to_string(kTargetVersion) + ")");
  }

  size_t off = sizeof(TargetFileHeader);
  positions_.reserve(header_.num_positions);
  for (uint32_t p = 0; p < header_.num_positions; ++p) {
    if (off + sizeof(TargetPositionHeader) > buffer_.size()) {
      throw std::runtime_error("TargetReader: truncated position header in " + path);
    }
    const TargetPositionHeader* ph =
      reinterpret_cast<const TargetPositionHeader*>(buffer_.data() + off);
    off += sizeof(TargetPositionHeader);
    const size_t bytes = static_cast<size_t>(ph->num_candidates) * record_bytes();
    if (off + bytes > buffer_.size()) {
      throw std::runtime_error("TargetReader: truncated records in " + path);
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

}  // namespace move_set_eval
}  // namespace scribblez
