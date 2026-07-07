#include "scribblez/mpre_target_log.h"

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace scribblez {

namespace {

void append_bytes(std::vector<char>* buffer, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  buffer->insert(buffer->end(), p, p + size);
}

}  // namespace

MpreTargetWriter::MpreTargetWriter(const std::string& path, uint32_t record_floats,
                                   const std::string& model_hash, uint32_t flags)
    : path_(path), record_floats_(record_floats) {
  MpreTargetFileHeader hdr{};
  hdr.magic = kMpreTargetMagic;
  hdr.version = kMpreTargetVersion;
  hdr.num_positions = 0;  // patched in close()
  hdr.record_floats = record_floats;
  hdr.flags = flags;
  std::strncpy(hdr.model_hash, model_hash.c_str(), sizeof(hdr.model_hash) - 1);
  append_bytes(&buffer_, &hdr, sizeof(hdr));
}

MpreTargetWriter::~MpreTargetWriter() {
  if (!closed_) close();
}

void MpreTargetWriter::add_position(uint32_t game_index, uint32_t turn_index,
                                    const std::vector<Move>& candidates,
                                    const std::vector<float>& targets) {
  assert(!closed_);
  assert(targets.size() == candidates.size() * record_floats_);
  MpreTargetPositionHeader ph{};
  ph.game_index = game_index;
  ph.turn_index = turn_index;
  ph.num_candidates = static_cast<uint32_t>(candidates.size());
  append_bytes(&buffer_, &ph, sizeof(ph));
  for (size_t c = 0; c < candidates.size(); ++c) {
    append_bytes(&buffer_, &candidates[c], sizeof(Move));
    append_bytes(&buffer_, targets.data() + c * record_floats_, sizeof(float) * record_floats_);
  }
  ++num_positions_;
}

void MpreTargetWriter::close() {
  assert(!closed_);
  closed_ = true;
  MpreTargetFileHeader* hdr = reinterpret_cast<MpreTargetFileHeader*>(buffer_.data());
  hdr->num_positions = num_positions_;
  // Temp-file + rename so the .mpt appears atomically: an interrupted run
  // never leaves a truncated file that a resume (which skips existing
  // sidecars) would silently keep.
  const std::string tmp = path_ + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("MpreTargetWriter: cannot open " + tmp);
    f.write(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
  }
  std::filesystem::rename(tmp, path_);
}

MpreTargetReader::MpreTargetReader(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("MpreTargetReader: cannot open " + path);
  const std::streamsize size = f.tellg();
  f.seekg(0);
  buffer_.resize(static_cast<size_t>(size));
  f.read(buffer_.data(), size);

  if (buffer_.size() < sizeof(MpreTargetFileHeader)) {
    throw std::runtime_error("MpreTargetReader: truncated header in " + path);
  }
  std::memcpy(&header_, buffer_.data(), sizeof(header_));
  if (header_.magic != kMpreTargetMagic) {
    throw std::runtime_error("MpreTargetReader: bad magic in " + path);
  }
  if (header_.version != kMpreTargetVersion) {
    throw std::runtime_error("MpreTargetReader: version mismatch in " + path +
                             " (file=" + std::to_string(header_.version) +
                             " code=" + std::to_string(kMpreTargetVersion) + ")");
  }

  size_t off = sizeof(MpreTargetFileHeader);
  positions_.reserve(header_.num_positions);
  for (uint32_t p = 0; p < header_.num_positions; ++p) {
    if (off + sizeof(MpreTargetPositionHeader) > buffer_.size()) {
      throw std::runtime_error("MpreTargetReader: truncated position header in " + path);
    }
    const MpreTargetPositionHeader* ph =
      reinterpret_cast<const MpreTargetPositionHeader*>(buffer_.data() + off);
    off += sizeof(MpreTargetPositionHeader);
    const size_t bytes = static_cast<size_t>(ph->num_candidates) * record_bytes();
    if (off + bytes > buffer_.size()) {
      throw std::runtime_error("MpreTargetReader: truncated records in " + path);
    }
    positions_.push_back({ph, buffer_.data() + off});
    off += bytes;
  }
}

std::string MpreTargetReader::model_hash() const {
  return std::string(header_.model_hash, strnlen(header_.model_hash, sizeof(header_.model_hash)));
}

Move MpreTargetReader::move_at(const Position& p, int candidate) const {
  Move m;
  std::memcpy(&m, p.records + candidate * record_bytes(), sizeof(Move));
  return m;
}

const float* MpreTargetReader::targets_at(const Position& p, int candidate) const {
  return reinterpret_cast<const float*>(p.records + candidate * record_bytes() + sizeof(Move));
}

}  // namespace scribblez
