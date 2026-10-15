#pragma once

#include <span>
#include <string>

namespace scribblez::util {

// A 16-digit hex FNV-1a 64-bit fingerprint of `bytes`, for recording which
// exact file (a model export, a leave-value table) produced some output.
std::string content_hash(std::span<const char> bytes);

// content_hash over the bytes of the file at `path`. Throws
// util::CleanException if the file cannot be opened.
std::string file_content_hash(const std::string& path);

}  // namespace scribblez::util
