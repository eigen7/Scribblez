#pragma once

// What the TensorRT parity binaries (py/tools/gpu_parity.py) share: reading a
// fixture's raw arrays, and generating a fixture with its Python generator
// into a scratch directory when no fixture directory is given.

#include <gtest/gtest.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace scribblez::testing {

// The file's bytes as an array of T. Records a test failure and returns an
// empty array if it cannot be opened.
template <typename T>
std::vector<T> read_binary(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    ADD_FAILURE() << "cannot open " << path;
    return {};
  }
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<T> out(size_t(bytes) / sizeof(T));
  if (bytes) f.read(reinterpret_cast<char*>(out.data()), bytes);
  return out;
}

// A fresh directory under the system temp dir, named from `prefix`.
inline std::filesystem::path make_scratch_dir(const char* prefix) {
  std::filesystem::path base =
    std::filesystem::temp_directory_path() /
    (prefix + std::to_string(::time(nullptr)) + "_" + std::to_string(std::random_device{}()));
  std::filesystem::create_directories(base);
  return base;
}

// Runs `python3 -m <module> --out-dir <out_dir> <extra_args>` from the py/
// directory at SCRIBBLEZ_PY_DIR. False if it fails (for example, torch or onnx
// is not installed) or the binary was built without SCRIBBLEZ_PY_DIR.
inline bool generate_fixture(const std::string& module, const std::string& out_dir,
                             const std::string& extra_args = "") {
#ifdef SCRIBBLEZ_PY_DIR
  const std::string py_dir = SCRIBBLEZ_PY_DIR;
  const std::string cmd = "cd \"" + py_dir + "\" && PYTHONPATH=\"" + py_dir + "\" python3 -m " +
                          module + " --out-dir \"" + out_dir + "\" " + extra_args;
  return std::system(cmd.c_str()) == 0;
#else
  (void)module;
  (void)out_dir;
  (void)extra_args;
  return false;
#endif
}

}  // namespace scribblez::testing
