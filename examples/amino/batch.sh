#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=2G
#SBATCH --output=logs/amino_%j_stdout.txt
#SBATCH --error=logs/amino_%j_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=amino_acid
#SBATCH --mail-user=YOUR EMAIL ADDRESS HERE
#SBATCH --mail-type=ALL
#SBATCH --chdir=YOUR DIRECTORY HERE

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up 

. /home/fagg/tf_setup.sh
conda activate dnn

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src


####

python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network.txt -vvv --force
