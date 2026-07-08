#include "util/cli.h"

#include "util/exception.h"

#include <boost/program_options.hpp>

#include <iostream>

namespace scribblez {

void parse_command_line(int argc, char** argv, boost::program_options::options_description& desc,
                        const std::string& help_epilog) {
  namespace po = boost::program_options;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
    throw Exception(e.what());
  }
  if (vm.count("help")) {
    std::cout << desc << "\n";
    if (!help_epilog.empty()) std::cout << help_epilog;
    throw CleanExit();
  }
}

}  // namespace scribblez
