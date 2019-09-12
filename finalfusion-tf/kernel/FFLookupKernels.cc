#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.h"

#include "finalfusion-cxx/Embeddings.hh"
#include "finalfusion-tf/FFLookup.hh"

using namespace tensorflow;

REGISTER_KERNEL_BUILDER(
    Name("FFEmbeddings").Device(DEVICE_CPU),
    ResourceHandleOp<FFLookup>);

class InitializeFFEmbeddingsOp : public OpKernel {
public:
  explicit InitializeFFEmbeddingsOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    FFLookup *lookup;
    // verbosely fail if the lookup has been initialized before
    bool const found = LookupResource(context, HandleFromInput(context, 0), &lookup).ok();
    if (found) {
      core::ScopedUnref unref(lookup);
      context->CtxFailure(errors::AlreadyExists("Lookup has already been created."));
    }

    Tensor const *tmp;
    OP_REQUIRES_OK(context, context->input("filename", &tmp));
    string const path = tmp->scalar<string>()();
    OP_REQUIRES_OK(context, context->input("mmap", &tmp));
    bool const mmap = tmp->scalar<bool>()();

    OP_REQUIRES_OK(context, LookupOrCreateResource<FFLookup>(
        context, HandleFromInput(context, 0), &lookup,
        [path, mmap](FFLookup **lookup) {
          return CreateFFLookup(path, mmap, lookup);
        }));
    core::ScopedUnref unref(lookup);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("InitializeFFEmbeddings").Device(DEVICE_CPU),
    InitializeFFEmbeddingsOp);

class CloseFFEmbeddingsOp : public OpKernel {
public:
  explicit CloseFFEmbeddingsOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    OP_REQUIRES_OK(context, context->resource_manager()->Delete(HandleFromInput(context, 0)));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CloseFFEmbeddings").Device(DEVICE_CPU), CloseFFEmbeddingsOp);
