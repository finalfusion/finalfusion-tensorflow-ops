from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pkg_resources import resource_filename
import platform

from tensorflow.python.framework import load_library

if platform.system() == "Darwin":
    LIB_SUFFIX = ".dylib"
else:
    LIB_SUFFIX = ".so"

_finalfusion_ops = load_library.load_op_library(resource_filename(__name__, "libfinalfusion_tf" + LIB_SUFFIX))


def ff_embeddings(container="", shared_name="", name=None):
    return _finalfusion_ops.ff_embeddings(container=container, shared_name=shared_name, name=name)


def initialize_ff_embeddings(embeddings, path="", mmap=False, name=None):
    return _finalfusion_ops.initialize_ff_embeddings(embeddings, path, mmap, name=name)


def ff_lookup(embeddings, query, embedding_len=-1, mask_empty_string=False, mask_failed_lookup=False, name=None):
    return _finalfusion_ops.ff_lookup(embeddings, query, embedding_len, mask_empty_string, mask_failed_lookup,
                                      name=name)


def close_ff_embeddings(embeddings, name=None):
    return _finalfusion_ops.close_ff_embeddings(embeddings, name=name)
