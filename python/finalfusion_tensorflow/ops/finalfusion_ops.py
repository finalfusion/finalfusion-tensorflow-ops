from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

if platform.system() == "Darwin":
    LIB_SUFFIX = ".dylib"
else:
    LIB_SUFFIX = ".so"

finalfusion_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('./libfinalfusion_tf' + LIB_SUFFIX))

ff_embeddings = finalfusion_ops.ff_embeddings
initialize_ff_embeddings = finalfusion_ops.initialize_ff_embeddings
ff_lookup = finalfusion_ops.ff_lookup
close_ff_embeddings = finalfusion_ops.close_ff_embeddings
