// leave_table_tool: writes the leave table (.ltab, belief/leave_table.h) for a
// lexicon's leave values. The leave inference trainer runs it at startup on
// its own machine.
//
//   leave_table_tool --out /tmp/NWL23.ltab
//   leave_table_tool --lexicon CSW24 --out /tmp/CSW24.ltab

#include "belief/leave_table.h"
#include "lexicon/hasty_equity.h"
#include "lexicon/lexicon.h"
#include "util/exception.h"
#include "util/misc.h"

#include <boost/program_options.hpp>

#include <chrono>
#include <format>
#include <iostream>
#include <string>

namespace {

using namespace scribblez;

void print_summary(const belief::LeaveTable& table, const std::string& out, double seconds) {
  for (int k = 1; k <= belief::kLeaveTableMaxLength; ++k) {
    std::cout << std::format("length {}: {} rows\n", k, table.rows(k).size());
  }
  std::cout << std::format("wrote {} ({}, klv {}) content id {} in {:.2f}s\n", out, table.lexicon(),
                           table.klv_hash(), table.content_id(), seconds);
}

}  // namespace

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  try {
    std::string out;
    po::options_description desc("leave_table_tool options");
    desc.add_options()("help,h", "show this help and exit")(
      "out", po::value<std::string>(&out)->required(), "path of the .ltab file to write");
    Lexicon::instance().add_options(desc);
    util::parse_command_line(argc, argv, desc);

    const auto start = std::chrono::steady_clock::now();
    const std::string& lexicon = Lexicon::instance().name();
    HastyEquity::ensure_initialized(lexicon);
    const belief::LeaveTable table = belief::LeaveTable::build(HastyEquity::instance(), lexicon);
    table.write(out);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    print_summary(table, out, elapsed.count());
    return 0;
  } catch (...) {
    return util::main_exit_code();
  }
}
