#!/bin/bash

set -euxo pipefail

python setup.py bdist_wheel
pip install dist/*.whl
python -c "import finalfusion_tensorflow"