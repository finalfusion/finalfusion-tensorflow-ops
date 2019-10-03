#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

  REGISTER_OP("FFEmbeddings")
    .Output("lookup: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetShapeFn(shape_inference::ScalarShape);

  REGISTER_OP("InitializeFFEmbeddings")
    .Input("embeds: resource")
    .Input("filename: string")
    .Input("mmap: bool")
    .SetShapeFn([](
        ::tensorflow::shape_inference::InferenceContext *c
    ) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &input));
      return Status::OK();
    });


  REGISTER_OP("CloseFFEmbeddings")
    .Input("embeds: resource")
    .SetShapeFn([](
        ::tensorflow::shape_inference::InferenceContext *c
    ) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input));
      return Status::OK();
    });

  REGISTER_OP("FFLookup")
    .Input("embeds: resource")
    .Input("query: string")
    .Attr("embedding_len: int >= -1 = -1")
    .Attr("mask_empty_string: bool = true")
    .Attr("mask_failed_lookup: bool = true")
    .Output("embeddings: float")
    .SetShapeFn([](
      ::tensorflow::shape_inference::InferenceContext *c
    ) {
      ShapeHandle embeds = c->output(0);
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &embeds));
      ShapeHandle strings_shape = c->input(1);
      ShapeHandle output_shape;
      int embedding_len;
      TF_RETURN_IF_ERROR(c->GetAttr("embedding_len", &embedding_len));
      TF_RETURN_IF_ERROR(
        c->Concatenate(strings_shape, c->Vector(embedding_len), &output_shape)
      );
      c->set_output(0, output_shape);
      return Status::OK();
    });
}
