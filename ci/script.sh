#!/bin/bash

set -euxo pipefail

mkdir build
cd build
cmake ..
make

# Verify that we can load the shared library in tensorflow.
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  python3 -c "import tensorflow as tf; tf.load_op_library('finalfusion-tf/libfinalfusion_tf.dylib')"
fi

if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  python3.6 -c "import tensorflow as tf; tf.load_op_library('finalfusion-tf/libfinalfusion_tf.so')"
fi
