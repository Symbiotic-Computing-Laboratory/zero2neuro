#!/bin/bash

# Test use of tf-datasets for amino acid case: 10 files, 5 folds, n-fold cross-validation; combining pairs of files to form the 5 folds
#  These tf-datasets are created by testH

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/amino

python $ZERO2NEURO_PATH/zero2neuro.py @data_tf3.txt @experiment.txt @network.txt  -vvv --force --epochs 2
