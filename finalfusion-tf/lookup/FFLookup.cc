#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

#include "finalfusion-tf/FFLookup.hh"

FFLookup::FFLookup(std::string const &path, bool mmap)
    : ResourceBase(),
      embeddings_(path, mmap) {}

FFLookup::~FFLookup() = default;

std::vector<float> FFLookup::embedding(std::string const &word) {
  return embeddings_.embedding(word);
}

size_t FFLookup::dimensions() {
  return embeddings_.dimensions();
}

std::string FFLookup::DebugString() const {
  return "FFLookup";
}

tensorflow::Status CreateFFLookup(std::string const &path, const bool mmap, FFLookup **result) {
  try {
    FFLookup *ff_lookup = new FFLookup(path, mmap);
    *result = ff_lookup;
  } catch (std::exception &e) {
    return tensorflow::errors::Unknown(e.what());
  }

  return tensorflow::Status::OK();
}
