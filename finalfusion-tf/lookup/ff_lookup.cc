#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include "finalfusion-tf/ff_lookup.hh"

using namespace tensorflow;

FFLookup::FFLookup(Env *env)
    : ResourceBase(),
      is_initialized_(false),
      env_(env) {}

Status FFLookup::Initialize(string const &path, bool mmap) {
  try {
    embeddings_ = std::make_unique<Embeddings>(path, mmap);
  } catch (std::exception &e) {
    return errors::Unknown(e.what());
  }

  is_initialized_ = true;
  return Status::OK();
}

Status FFLookup::Close() {
  if (!is_initialized_) {
    return errors::FailedPrecondition("Can't close uninitialized lookup.");
  }
  embeddings_.reset(NULL);
  return Status::OK();
}

Status FFLookup::embedding(const string &word, std::vector<float> *result) {
  *result = embeddings_->embedding(word);
  if (result->empty()) {
    return errors::InvalidArgument("Could not retrieve an embedding for: ", word);
  }
  return Status::OK();
}

size_t FFLookup::dimensions() {
  CHECK_NOTNULL(embeddings_);
  return embeddings_->dimensions();
}

bool FFLookup::initialized() {
  return is_initialized_;
}

string FFLookup::DebugString() const {
  return "FFLookup";
}

/// Creates a new initialized FiFuLookup.
Status CreateFFLookup(const string &path, const bool mmap, Env *env, FFLookup **result) {
  FFLookup *ff_lookup = new FFLookup(env);
  const Status init = ff_lookup->Initialize(path, mmap);
  if (!init.ok()) {
    ff_lookup->Unref();
    *result = nullptr;
    return init;
  }
  *result = ff_lookup;

  return Status::OK();
}