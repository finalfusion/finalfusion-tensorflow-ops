#ifndef FF_LOOKUP_H
#define FF_LOOKUP_H
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/resource_mgr.h"

#include "finalfusion-cxx/Embeddings.hh"

using namespace tensorflow;

class FFLookup : public ResourceBase {
public:
  FFLookup(Env *env);
  Status Initialize(string const &filename, bool mmap);
  Status Close();
  Status embedding(const string &word, std::vector<float> *result);
  size_t dimensions();
  bool initialized();
  string DebugString() const override;
private:
  bool is_initialized_;
  std::unique_ptr<Embeddings> embeddings_;
  Env *env_;
};

Status CreateFFLookup(string const &path, bool mmap, Env *env, FFLookup **result);
#endif //FF_LOOKUP_H
