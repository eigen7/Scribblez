// The web tools' shared serving pieces: ViteDevServer's startup failure and
// the bag JSON the tool UIs render (serve/position_json.h).

#include "game/tile.h"
#include "game/tile_counts.h"
#include "serve/position_json.h"
#include "serve/web_server.h"
#include "util/exception.h"

#include <arpa/inet.h>
#include <boost/json.hpp>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <filesystem>
#include <string>
#include <unistd.h>

namespace scribblez {
namespace {

// A loopback port nothing listens on: the kernel's pick for a socket bound to
// port 0, released before returning. ViteDevServer kills whatever listens on
// its port, so the test must not borrow one in use.
int free_port() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
  ::close(fd);
  return ntohs(addr.sin_port);
}

TEST(ViteDevServer, StartupFailureNamesTheLogInTheWebDir) {
  // No package.json here or above, so `npm run dev` exits at once.
  const std::filesystem::path web_dir =
    std::filesystem::temp_directory_path() / ("scribblez_vite_" + std::to_string(::getpid()));
  std::filesystem::create_directories(web_dir);
  const std::string log = (web_dir / ".vite-dev.log").string();

  ViteDevServer vite(web_dir.string(), free_port(), free_port(), "", "web", 5173);
  try {
    vite.wait_until_ready(/*timeout_ms=*/20000);
    ADD_FAILURE() << "expected the dev server to fail to start";
  } catch (const util::CleanException& e) {
    EXPECT_NE(std::string(e.what()).find(log), std::string::npos) << e.what();
  }
  std::filesystem::remove_all(web_dir);
}

TEST(PositionJson, BagTilesListsLettersThenBlankAndSkipsEmpty) {
  TileCounts bag;
  bag.add(Tile::from_char('Q'));
  bag.add(Tile::from_char('A'));
  bag.add(Tile::from_char('A'));
  bag.add(BLANK);
  EXPECT_EQ(boost::json::serialize(bag_tiles_json(bag)),
            R"([{"letter":"A","score":1,"count":2},{"letter":"Q","score":10,"count":1},)"
            R"({"letter":"?","score":0,"count":1}])");
}

}  // namespace
}  // namespace scribblez
