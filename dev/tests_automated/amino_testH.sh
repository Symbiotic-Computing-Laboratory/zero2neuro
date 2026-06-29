#!/bin/bash

# Test random-stratify assignment to tf-dataset folds

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/amino

python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network.txt  -vvv --data_save_folds ds/amino --data_representation tf-dataset --nogo
