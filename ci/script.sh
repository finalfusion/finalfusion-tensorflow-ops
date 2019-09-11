#!/bin/bash

set -euxo pipefail

mkdir build
cd build
cmake ..
make

python3.6 -c "import tensorflow as tf; tf.load_op_library('finalfusion-tf/libfinalfusion_tf_op.so')
