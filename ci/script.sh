#!/bin/bash

set -euxo pipefail

python setup.py build_ext --test install --verbose

pushd tests
pytest test_eager_mode.py
pytest test_graph_mode.py
popd