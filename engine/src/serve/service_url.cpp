#include "serve/service_url.h"

#include <cctype>
#include <cstdlib>
#include <format>
#include <string>

namespace scribblez {

namespace {

// The DEVENV_SERVICE_URL_<NAME> variable name for a service: uppercased, with
// '-' mapped to '_', matching the contract run_docker.py exports.
std::string env_var_name(const std::string& service) {
  std::string name = "DEVENV_SERVICE_URL_";
  for (char c : service) {
    name.push_back(c == '-' ? '_' : static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return name;
}

}  // namespace

std::string service_url(const std::string& service, int port, int default_port) {
  const char* url = std::getenv(env_var_name(service).c_str());
  if (url != nullptr && port == default_port) return url;
  return std::format("http://localhost:{}", port);
}

}  // namespace scribblez
