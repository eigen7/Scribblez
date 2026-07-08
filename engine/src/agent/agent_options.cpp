#include "agent/agent_options.h"

#include <sstream>

namespace scribblez {

std::string agent_options_help(const std::string& description,
                               const boost::program_options::options_description& opts) {
  std::ostringstream o;
  o << description;
  if (opts.options().empty()) {
    o << "  Options: (none)\n";
  } else {
    o << "  Options:\n" << opts;
  }
  return o.str();
}

}  // namespace scribblez
