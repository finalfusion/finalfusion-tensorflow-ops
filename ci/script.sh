#!/bin/bash

set -euxo pipefail

mkdir build
cd build
cmake ..
make
