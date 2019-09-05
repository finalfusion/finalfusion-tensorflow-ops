#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/resource_handle.h"

#include "finalfusion-cxx/Embeddings.hh"
#include "finalfusion-tf/ff_lookup.hh"

REGISTER_KERNEL_BUILDER(
    Name("FFEmbeddings").Device(DEVICE_CPU),
    ResourceHandleOp<FFLookup>);

class InitializeFFEmbeddingsOp : public OpKernel {
public:
  explicit InitializeFFEmbeddingsOp(OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    FFLookup *s;
    // verbosely fail if the lookup has been initialized before
    bool found = LookupResource(context, HandleFromInput(context, 0), &s).ok();
    if (found) {
      core::ScopedUnref unref(s);
      context->CtxFailure(errors::AlreadyExists("Lookup has already been created."));
    }

    const Tensor *tmp;
    OP_REQUIRES_OK(context, context->input("filename", &tmp));
    const string path = tmp->scalar<string>()();
    OP_REQUIRES_OK(context, context->input("mmap", &tmp));
    const bool mmap = tmp->scalar<bool>()();

    OP_REQUIRES_OK(context, LookupOrCreateResource<FFLookup>(
        context, HandleFromInput(context, 0), &s,
        [path, mmap, context](FFLookup **s) {
          return CreateFFLookup(path, mmap, context->env(), s);
        }));
    core::ScopedUnref unref(s);
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

class FFLookupOp : public OpKernel {
public:
  explicit FFLookupOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mask_empty_string", &mask_empty_string_));
    OP_REQUIRES_OK(context, context->GetAttr("mask_failed_lookup", &mask_failed_lookup_));
  }

  void Compute(OpKernelContext *context) override {
    FFLookup *e;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &e));
    core::ScopedUnref unref(e);
    OP_REQUIRES(context, e->initialized(), errors::FailedPrecondition("Class was not properly initialized."));
    size_t embedding_dims = e->dimensions();
    // Grab the input tensor
    const Tensor &input_tensor = context->input(1);
    TensorShape out_shape(input_tensor.shape());
    out_shape.AddDim(((int64) embedding_dims));
    auto input = input_tensor.flat<string>();
    // Create an output tensor
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    auto n = input.size();
    for (int i = 0; i < n; i++) {
      std::vector<float> embed;
      Status s = e->embedding(input(i), &embed);
      if ((input(i).empty() && mask_empty_string_) || (mask_failed_lookup_ && !s.ok())) {
        std::memset(&output_flat(i * embedding_dims), 0, embedding_dims * 4);
      } else {
        OP_REQUIRES(context, s.ok(), s);
        std::memcpy(&output_flat(i * embedding_dims), embed.data(), embedding_dims * sizeof(float));
      }
    }
  }
private:
  bool mask_empty_string_;
  bool mask_failed_lookup_;
};

REGISTER_KERNEL_BUILDER(
    Name("FFLookup").Device(DEVICE_CPU),
    FFLookupOp);
