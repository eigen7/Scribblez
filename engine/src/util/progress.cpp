#include "util/progress.h"

#include <algorithm>
#include <iostream>
#include <unistd.h>

namespace scribblez::util {

ProgressMeter::ProgressMeter(uint64_t total, std::string noun)
    : total_(total), noun_(std::move(noun)), start_(std::chrono::steady_clock::now()) {
  monitor_ = std::thread(&ProgressMeter::monitor_loop, this);
}

ProgressMeter::~ProgressMeter() { stop_monitor(); }

double ProgressMeter::elapsed_s() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
}

void ProgressMeter::render() const {
  if (!::isatty(2) || total_ == 0) return;
  const uint64_t d = done();
  constexpr int kWidth = 40;
  const int filled = int(kWidth * d / total_);
  std::string bar(size_t(filled), '#');
  bar.resize(kWidth, '.');
  const double rate = d / std::max(elapsed_s(), 1e-9);
  const double eta = rate > 0 ? double(total_ - d) / rate : 0.0;
  std::cerr << "\r  [" << bar << "] " << d << "/" << total_ << " " << noun_ << " ("
            << int(100 * d / total_) << "%, eta " << int(eta) << "s)  " << std::flush;
}

void ProgressMeter::monitor_loop() {
  while (!stop_.load(std::memory_order_relaxed)) {
    render();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

void ProgressMeter::stop_monitor() {
  if (!monitor_.joinable()) return;
  stop_.store(true, std::memory_order_relaxed);
  monitor_.join();
}

void ProgressMeter::finish(const std::string& prefix) {
  stop_monitor();
  if (::isatty(2)) std::cerr << "\r\033[K";  // clear the bar line
  const double s = elapsed_s();
  std::cerr << prefix << ": " << done() << " " << noun_ << " in " << int(s) << "s ("
            << (done() / std::max(s, 1e-9)) << " " << noun_ << "/s)\n";
}

}  // namespace scribblez::util
