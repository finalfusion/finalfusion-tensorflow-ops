#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
    Tensor const *tmp;
    OP_REQUIRES_OK(context, context->input("embeds", &tmp));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("'embeds' must be of rank 0, not: ", tmp->shape().dims())
    );
    ResourceHandle embed_handle = tmp->flat<ResourceHandle>()(0);

    FFLookup *lookup;
    // verbosely fail if the lookup has been initialized before
    bool const found = LookupResource(context, embed_handle, &lookup).ok();
    if (found) {
      core::ScopedUnref unref(lookup);
      context->CtxFailure(errors::AlreadyExists("Lookup has already been created."));
    }

    OP_REQUIRES_OK(context, context->input("filename", &tmp));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("'filename' must be of rank 0, not: ",
                                        tmp->shape().dims())
    );
    string const path = tmp->scalar<string>()();

    OP_REQUIRES_OK(context, context->input("mmap", &tmp));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("'mmap' must be of rank 0, not: ",
                                        tmp->shape().dims())
    );
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
    Tensor const *tmp;
    OP_REQUIRES_OK(context, context->input("embeds", &tmp));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("'embeds' must be of rank 0, not: ", tmp->shape().dims())
    );
    ResourceHandle embed_handle = tmp->flat<ResourceHandle>()(0);
    OP_REQUIRES_OK(context, context->resource_manager()->Delete(embed_handle));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("CloseFFEmbeddings").Device(DEVICE_CPU),
    CloseFFEmbeddingsOp);

class FFLookupOp : public OpKernel {
public:
  explicit FFLookupOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mask_empty_string", &mask_empty_string_));
    OP_REQUIRES_OK(context, context->GetAttr("mask_failed_lookup", &mask_failed_lookup_));
    OP_REQUIRES_OK(context, context->GetAttr("embedding_len", &embedding_len_));
  }

  void Compute(OpKernelContext *context) override {
    Tensor const *tmp;
    OP_REQUIRES_OK(context, context->input("embeds", &tmp));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(tmp->shape()),
                errors::InvalidArgument("'embeds' must be of rank 0, not: ", tmp->shape().dims())
    );
    ResourceHandle embed_handle = tmp->flat<ResourceHandle>()(0);
    FFLookup *lookup;
    OP_REQUIRES_OK(context, LookupResource(context, embed_handle, &lookup));
    core::ScopedUnref unref(lookup);

    // verify length from construction with actual length
    size_t const dims = lookup->dimensions();
    if (embedding_len_ != -1) {
      OP_REQUIRES(context,
                  (dims == embedding_len_),
                  errors::InvalidArgument("Actual embedding length (", dims, ") does not match provided length (",
                                          embedding_len_, ")"));
    }

    // Get input tensor and flatten
    Tensor const &query_tensor = context->input(1);
    auto query = query_tensor.flat<string>();

    // Set output shape: add new dim with dimensionality of embeddings
    TensorShape out_shape(query_tensor.shape());
    out_shape.AddDim(((int64) dims));

    // Create output tensor and flatten
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    for (int i = 0; i < query.size(); i++) {
      std::vector<float> embedding = lookup->embedding(query(i));
      // optionally mask failed lookups and/or empty string. Generally, empty string will lead to a failed lookup.
      if ((query(i).empty() && mask_empty_string_) || (mask_failed_lookup_ && embedding.empty())) {
        std::memset(&output_flat(i * dims), 0., dims * sizeof(float));
      } else {
        // if no masking attributes are set and the embedding is empty, return error.
        OP_REQUIRES(context, !embedding.empty(), errors::InvalidArgument("Embedding lookup failed for: ", query(i)));
        std::memcpy(&output_flat(i * dims), embedding.data(), dims * sizeof(float));
      }
    }
  }

private:
  bool mask_empty_string_;
  bool mask_failed_lookup_;
  int embedding_len_;
};

REGISTER_KERNEL_BUILDER(
    Name("FFLookup").Device(DEVICE_CPU),
    FFLookupOp);