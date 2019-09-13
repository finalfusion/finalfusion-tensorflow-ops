import os
import platform

import pytest
import tensorflow as tf

tf.enable_eager_execution()


@pytest.fixture
def ff_lib(tests_root):
    if platform.system() == "Darwin":
        LIB_SUFFIX = ".dylib"
    else:
        LIB_SUFFIX = ".so"

    yield tf.load_op_library("./finalfusion-tf/libfinalfusion_tf" + LIB_SUFFIX)


@pytest.fixture
def tests_root():
    yield os.path.dirname(__file__)