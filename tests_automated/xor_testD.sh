#!/bin/bash
# Scikit-Learn test

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/xor

python $ZERO2NEURO_PATH/zero2neuro.py @network_skl3.txt @data.txt @experiment.txt -vvv --force

