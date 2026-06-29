#!/bin/bash

# Test use of tf-datasets for amino acid case
#  These tf-datasets are created by testH

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/amino

python $ZERO2NEURO_PATH/zero2neuro.py @data_tf.txt @experiment.txt @network.txt  -vvv --force --epochs 2
