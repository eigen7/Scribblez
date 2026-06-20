// Off-line comparison of two estimators of the eventual score differential at
// the sampled position of each game in a .slog dataset:
//   * HastyBot static equity of the played move, and
//   * the value model's post-move score_diff_mean.
// against the realized final differential. For each sampled position it emits
// one CSV row (to stdout); a companion script (py/scripts/value_vs_equity.py)
// runs the regressions and reports which estimator better predicts outcomes.
//
// Run it on the FROZEN TEST split only -- the model trained on train/, so its
// predictions there are optimistically biased and the comparison would be
// rigged in the model's favor.
//
// Usage:
//   value_vs_equity <model.onnx> <slog_dir> [--lexicon=NWL23] [--precision=FP16]
//                   [--batch=256] [--max=N] > rows.csv
//
// Diagnostics go to stderr so stdout stays clean CSV.

#include "scribblez/binary_log.h"
#include "scribblez/block_decoder.h"
#include "scribblez/hasty_equity.h"
#include "scribblez/input_encoder.h"
#include "scribblez/nn/nn_evaluation_service.h"
#include "scribblez/nn/trt_util.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using scribblez::HastyEquity;
using scribblez::binlog::BlockDecoder;
using scribblez::binlog::FileHeader;
using scribblez::binlog::kInputFloats;
using scribblez::binlog::kMagic;
using scribblez::binlog::kVersion;
using scribblez::binlog::ValueProbe;
using scribblez::nn::Eval;
using scribblez::nn::NeuralNetParams;
using scribblez::nn::NNEvaluationService;

// One CSV record minus the model output (filled in at flush time, batched).
struct ProbeRow {
  int pov;
  int pre_move_diff;
  int bag_size;
  int final_diff;
  double equity;
};

// Read an entire file into a byte buffer.
std::vector<char> read_whole_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf(static_cast<size_t>(n));
  if (n > 0 && !f.read(buf.data(), n)) throw std::runtime_error("short read on " + path);
  return buf;
}

// Evaluate the accumulated batch of model inputs and print one CSV row per
// position, then clear the batch.
void flush_batch(NNEvaluationService& svc, const std::vector<float>& inputs,
                 std::vector<ProbeRow>& rows) {
  if (rows.empty()) return;
  std::vector<Eval> evals = svc.evaluate(inputs.data(), static_cast<int>(rows.size()));
  for (size_t i = 0; i < rows.size(); ++i) {
    const ProbeRow& r = rows[i];
    std::cout << r.pov << ',' << r.pre_move_diff << ',' << r.bag_size << ',' << r.equity << ','
              << evals[i].score_diff_mean << ',' << r.final_diff << '\n';
  }
  rows.clear();
}

// Pull "--key=value"; returns true and sets `out` if `arg` matches `key`.
bool match_opt(const std::string& arg, const std::string& key, std::string& out) {
  const std::string prefix = key + "=";
  if (arg.rfind(prefix, 0) != 0) return false;
  out = arg.substr(prefix.size());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  std::string data_dir;
  std::string lexicon = "NWL23";
  std::string precision = "FP16";
  int batch = 256;
  int max_positions = 0;  // 0 == no cap

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    std::string v;
    if (match_opt(a, "--lexicon", v)) {
      lexicon = v;
    } else if (match_opt(a, "--precision", v)) {
      precision = v;
    } else if (match_opt(a, "--batch", v)) {
      batch = std::atoi(v.c_str());
    } else if (match_opt(a, "--max", v)) {
      max_positions = std::atoi(v.c_str());
    } else {
      positional.push_back(a);
    }
  }
  if (positional.size() < 2 || batch < 1) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <slog_dir> [--lexicon=NWL23] [--precision=FP16|FP32]"
                 " [--batch=256] [--max=N]\n";
    return 1;
  }
  model = positional[0];
  data_dir = positional[1];

  try {
    HastyEquity::init(HastyEquity::default_leaves_path(lexicon), HastyEquity::default_peg_path());

    NeuralNetParams params;
    params.max_batch_size = batch;
    params.precision = scribblez::nn::parse_precision(precision);
    NNEvaluationService service(params);
    service.load(model);

    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(data_dir))
      if (e.path().extension() == ".slog") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    if (files.empty()) {
      std::cerr << "no .slog files in " << data_dir << "\n";
      return 1;
    }

    std::cout << "pov,pre_move_diff,bag_size,equity,score_diff_mean,final_diff\n";
    std::cout << std::setprecision(8);

    BlockDecoder dec;
    std::vector<float> inputs(static_cast<size_t>(batch) * kInputFloats);
    std::vector<ProbeRow> rows;
    rows.reserve(batch);
    int total = 0;
    bool capped = false;

    for (const std::string& path : files) {
      if (capped) break;
      std::vector<char> buf = read_whole_file(path);
      const FileHeader* hdr = reinterpret_cast<const FileHeader*>(buf.data());
      if (hdr->magic != kMagic || hdr->version != kVersion) {
        std::cerr << "skipping (bad magic/version): " << path << "\n";
        continue;
      }
      const uint32_t num_games = hdr->num_games;
      for (uint32_t g = 0; g < num_games; ++g) {
        if (max_positions > 0 && total >= max_positions) {
          capped = true;
          break;
        }
        float* dst = inputs.data() + rows.size() * kInputFloats;
        ValueProbe p = dec.value_probe(buf.data(), g, dst);
        const double equity = HastyEquity::instance().equity(p.played_move, p.board, p.bag_size,
                                                             p.opp_rack, p.mover_rack);
        rows.push_back({p.pov, p.pre_move_diff, p.bag_size, p.final_diff, equity});
        ++total;
        if (static_cast<int>(rows.size()) == batch) flush_batch(service, inputs, rows);
      }
    }
    flush_batch(service, inputs, rows);
    std::cerr << "wrote " << total << " positions\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "value_vs_equity error: " << ex.what() << "\n";
    return 1;
  }
}
