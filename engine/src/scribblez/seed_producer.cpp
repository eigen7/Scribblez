#include "scribblez/seed_producer.h"

#include <boost/program_options.hpp>

namespace scribblez {

SeedProducer::SeedProducer() {
  std::random_device rd;
  rng_.seed((static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()));
}

SeedProducer& SeedProducer::instance() {
  static SeedProducer p;
  return p;
}

void SeedProducer::Params::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()(
      "seed", po::value<uint64_t>(&seed)->default_value(seed),
      "PRNG seed (0 -- the default -- means pick one from std::random_device)");
}

uint64_t SeedProducer::seed(const Params& params) {
  uint64_t s = params.seed;
  if (s == 0) {
    std::random_device rd;
    s = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
  }
  rng_.seed(s);
  return s;
}

uint64_t SeedProducer::next() { return rng_(); }

}  // namespace scribblez
