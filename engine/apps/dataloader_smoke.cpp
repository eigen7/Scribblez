// dataloader_smoke: ad-hoc test driver for the DataLoader.
//
// Usage:
//   dataloader_smoke DIR [--samples N] [--workers W] [--prefetch P] [--budget MB]
//
// Walks DIR for *.slog files (in lexicographic == chronological order),
// registers them, then runs one load() over the full window and prints
// summary stats over the resulting rows.

#include "scribblez/binary_log.h"
#include "scribblez/data_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool read_header(const std::string& path, scribblez::binlog::FileHeader& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.read(reinterpret_cast<char*>(&out), sizeof(out));
  return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace scribblez::binlog;
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " DIR [--samples N] [--workers W] [--prefetch P] [--budget MB]\n";
    return 2;
  }
  const std::string dir = argv[1];
  int n_samples = 64;
  DataLoader::Params params;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--samples")
      n_samples = std::stoi(next());
    else if (a == "--workers")
      params.num_worker_threads = std::stoi(next());
    else if (a == "--prefetch")
      params.num_prefetch_threads = std::stoi(next());
    else if (a == "--budget-bytes")
      params.memory_budget = static_cast<int64_t>(std::stoll(next()));
    else if (a == "--budget")
      params.memory_budget = static_cast<int64_t>(std::stoll(next())) * 1024 * 1024;
    else {
      std::cerr << "unknown flag: " << a << "\n";
      return 2;
    }
  }

  // Discover .slog files in lexicographic (== timestamp == chronological) order.
  std::vector<std::filesystem::path> paths;
  for (auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.is_regular_file() && e.path().extension() == ".slog") {
      paths.push_back(e.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    std::cerr << "no .slog files in " << dir << "\n";
    return 1;
  }

  DataLoader loader(params);
  int64_t total_positions = 0;
  for (auto& p : paths) {
    FileHeader hdr{};
    if (!read_header(p.string(), hdr) || hdr.magic != kMagic) {
      std::cerr << "skipping malformed: " << p << "\n";
      continue;
    }
    const int64_t fsz = std::filesystem::file_size(p);
    loader.add_file(p.string(), hdr.num_positions, fsz);
    total_positions += hdr.num_positions;
    std::cout << "registered " << p.filename() << ": positions=" << hdr.num_positions
              << " bytes=" << fsz << "\n";
  }
  std::cout << "total positions across " << loader.num_files()
            << " files: " << loader.num_positions() << "\n";

  std::vector<float> out(static_cast<size_t>(n_samples) * DataLoader::row_size_floats());

  auto t0 = std::chrono::steady_clock::now();
  loader.load(0, loader.num_positions(), n_samples, /*apply_symmetry=*/false, out.data());
  auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();

  std::cout << "load(" << n_samples << ") in " << secs << "s (" << (n_samples / secs)
            << " rows/s); resident=" << loader.resident_bytes() << " B\n";

  // Summary: per-row WLD distribution and score-diff stats.
  int w = 0, d = 0, l = 0;
  double sd_sum = 0.0, sd_min = 1e9, sd_max = -1e9;
  const int RS = DataLoader::row_size_floats();
  for (int i = 0; i < n_samples; ++i) {
    const float* row = out.data() + i * RS;
    const float* wld = row + DataLoader::input_size_floats();
    if (wld[0] > 0.5f)
      ++w;
    else if (wld[1] > 0.5f)
      ++d;
    else
      ++l;
    const float sd = wld[3];
    sd_sum += sd;
    sd_min = std::min<double>(sd_min, sd);
    sd_max = std::max<double>(sd_max, sd);
  }
  std::cout << "WLD distribution: W=" << w << " D=" << d << " L=" << l << "\n";
  std::cout << "score_diff: mean=" << (sd_sum / n_samples) << " min=" << sd_min << " max=" << sd_max
            << "\n";

  // Show the label tail (last 4 floats) of the first 2 rows; the full 7000+
  // input floats are too noisy to dump.
  const int IS = DataLoader::input_size_floats();
  for (int i = 0; i < std::min(2, n_samples); ++i) {
    const float* row = out.data() + i * RS;
    std::cout << "row[" << i << "] labels:";
    for (int j = IS; j < RS; ++j) std::cout << " " << row[j];
    // Also report a few input-vector sanity numbers.
    int nonzero_spatial = 0;
    for (int j = 0; j < IS; ++j)
      if (row[j] != 0.0f) ++nonzero_spatial;
    std::cout << "  (nonzero input floats=" << nonzero_spatial << "/" << IS << ")\n";
  }
  return 0;
}
