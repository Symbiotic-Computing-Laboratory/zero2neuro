#!/bin/bash
#
#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=2G
#SBATCH --output=logs/xor_%j_stdout.txt
#SBATCH --error=logs/xor_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=xor_test
#SBATCH --mail-user=YOUR EMAIL ADDRESS HERE
#SBATCH --mail-type=ALL
#SBATCH --chdir=YOUR DIRECTORY HERE

#################################################

# Comment these lines in if running on the supercomputer
# . /home/fagg/tf_setup.sh
# conda activate dnn

# NEURO_REPOSITORY_PATH should already be defined

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

# Create the needed directories
mkdir -p results
mkdir -p logs

####

# Do the work
# Full training set
#python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config.txt @experiment_config.txt -vvv -dddd

# Partial training set with full data set as validation
#python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config2.txt @experiment_config.txt -vvv -dd

# Assemble full training set from parts with full data set as validation
#python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config3.txt @experiment_config.txt -vvv -dddd

# Assemble full training set from parts with full data set as validation.  
# Ins are in two parts in the pickle files & have to be  connected together
#python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config4.txt @experiment_config.txt -vvv -dddd

# Both tests at the same time
#python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config5.txt @experiment_config.txt -vvv -dddd

# --wandb --wandb_project 'parity'

# Both tests at the same time + testing holistic-cross-validation
python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config6.txt @experiment_config.txt -vvv -dddd --force

