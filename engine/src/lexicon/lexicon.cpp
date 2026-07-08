#include "lexicon/lexicon.h"

#include <boost/program_options.hpp>

#include <stdexcept>

namespace scribblez {

Lexicon& Lexicon::instance() {
  static Lexicon lex;
  return lex;
}

void Lexicon::add_options(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()  //
    ("lexicon", po::value<std::string>(&params_.name)->default_value(params_.name),
     "lexicon name; loaded from <lexica-dir>/<lexicon>.kwg. Run "
     "setup_wizard.py outside the Docker container to install lexica.")  //
    ("lexica-dir", po::value<std::string>(&params_.dir)->default_value(params_.dir),
     "directory holding .kwg files (rarely overridden)");
}

void Lexicon::set_params(const Params& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (dict_) {
    throw std::runtime_error("Lexicon::set_params called after dict() was loaded");
  }
  params_ = params;
}

const Dictionary& Lexicon::dict() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!dict_) {
    dict_ = std::make_unique<Dictionary>(Dictionary::load_kwg(kwg_path()));
  }
  return *dict_;
}

}  // namespace scribblez
