#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

  REGISTER_OP("FFEmbeddings")
    .Output("lookup: resource")
    .Attr("shared_name: string = ''")
    .Attr("container: string = ''")
    .SetShapeFn(shape_inference::NoOutputs);

  REGISTER_OP("InitializeFFEmbeddings")
    .Input("embeds: resource")
    .Input("filename: string")
    .Input("mmap: bool")
    .SetShapeFn(shape_inference::NoOutputs);


  REGISTER_OP("CloseFFEmbeddings")
    .Input("embeds: resource")
    .SetShapeFn(shape_inference::NoOutputs);
}
