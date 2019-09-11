#ifndef FF_LOOKUP_H
#define FF_LOOKUP_H
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

#include "finalfusion-cxx/Embeddings.hh"

class FFLookup : public tensorflow::ResourceBase {
public:
  /**
   * FFLookup Constructor.
   *
   * @param filename path to embeddings.
   * @param mmap memmap embeddings.
   * @throws runtime_error if Embeddings could not be read.
   */
  FFLookup(std::string const &filename, bool mmap);

  virtual ~FFLookup();

  /**
   * Embedding lookup
   * @param word the query word
   * @return the embedding. Empty if none could be found.
   */
  std::vector<float> embedding(std::string const &word);

  /// Return embedding dimensionality.
  size_t dimensions();

  std::string DebugString() const override;
private:
  Embeddings embeddings_;
};

/// Creates a new initialized FFLookup.
tensorflow::Status CreateFFLookup(std::string const &path, bool mmap, FFLookup **result);
#endif //FF_LOOKUP_H
