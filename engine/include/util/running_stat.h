#pragma once

#include <cmath>

namespace scribblez::util {

// Mean and variance of a stream of values by Welford's algorithm, which stays
// accurate where the naive sum-of-squares formula cancels catastrophically.
// Mirrors Macondo's stats.Statistic, so a port reproduces its sim statistics.
class RunningStat {
 public:
  void push(double x);

  int count() const { return n_; }
  double mean() const { return n_ > 0 ? mean_ : 0.0; }
  // Sample variance (n - 1 denominator); 0 below two values.
  double variance() const { return n_ > 1 ? m2_ / (n_ - 1) : 0.0; }
  // Standard error of the mean. NaN with no values, as in Macondo.
  double standard_error() const { return std::sqrt(variance() / n_); }

 private:
  int n_ = 0;
  double mean_ = 0.0;
  double m2_ = 0.0;  // sum of squared deviations from the running mean
};

}  // namespace scribblez::util

#include "inlines/util/running_stat.inl"
