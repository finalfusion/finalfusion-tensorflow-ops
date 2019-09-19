import platform

import pytest
import tensorflow as tf


@pytest.fixture
def ff_lib():
    if platform.system() == "Darwin":
        LIB_SUFFIX = ".dylib"
    else:
        LIB_SUFFIX = ".so"

    yield tf.load_op_library("./finalfusion-tf/libfinalfusion_tf" + LIB_SUFFIX)
