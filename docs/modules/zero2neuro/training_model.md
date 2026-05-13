---
title: Model Training
nav_order: 10
parent: Zero2Neuro Engine
has_children: false
---
  
# Model Training
Training a deep neural network is an incremental procedure.  Each of these _training epochs_ includes:
1. Evaluating the DNN with respect to the training data.
2. Estimating how to adjust each of the model parameters to slightly improve the model's performance (i.e., computing the error gradients).
3. Making small changes to the model parameters.

In addition, there exist a range of options that control how the model is trained, logged, and reported.

## Training Details
The heart of the training is driven by the [Keras Adam Optimizer](https://keras.io/api/optimizers/adam/).  The behavior of the optimizer is controlled by the following arguments:

- ```--loss``` specifies a string that defines the mechanism for measuring model prediction error (often referred to as _loss_ or _cost_).   The choice of loss function depends on the type of problem that you are solving.  Common choices include:
   - _mse_: Mean squared error: regression.
   - _mae_: Mean absolute error: regression.
   - _binary_crossentropy_: membership in one or more binary classes that do not interact with one-another.  Typically coupled with the _sigmoid_ activation function.
   - _categorical_crossentropy_ or _sparse_categorical_crossentopy_: membership in exactly one of N classes.  Typically coupled with the _softmax_ activation function.
   - Note: the specified loss will potentially be added to other losses, including some regularization losses (e.g., L1/L2 regularization).

- ```--metrics``` specifies a list off strings that describe the human-interpretable measures of model performance.  Common choices include:
   - _mse_ / _mae_: regression.
   - _binary_accuracy_: used in concert with _binary_crossentropy_.
   - _categorical_accuracy_ or _sparse_categorical_accuracy_: used in concert with _categorical_crossentropy_ or _sparse_categorical_crossentopy_, respectively

- ```epochs``` specifies an integer that determines the maximum number of model training epochs.  The choice of this argument depends dramatically on the problem types and network architecture.  Common choices range between the 100s to the 1000s (default: 100).

- ```--learning_rate``` specifies the real valued training "step size".  The choice of this argument also depends on the type of problem and network architecture.  Typically choices range between 0.01 and 0.0001 (default: 0.0001).

- ```--clipnorm``` (optional) specifies the real valued maximum gradient that the optimizer will use.  This effectively limits the magnitude of the change of model parameters for any given training step.  This argument is particularly useful with complex and deep models.  Typical choices are 0.01 to 0.0001 (default: None).

- ```--batch``` specifies an integer number of examples in a training batch (default: Keras chooses).  The choice of batch size depends greatly on the nature of the data and the model.  Reasonable heuristics:
   - Choose powers of 2.
   - Set as high as possible as long as the batch fits within GPU memory.


## Stopping Training Early
In many situations, the model training process will reach a stage where model performance will not improve with continued training (in fact, ```--epochs``` should be set so that this is the case).  _Early Stopping_ is a mechanism that attempts to detect when training has effectively completed by _monitoring_ a specified performance metric.  Once this monitored metric has reached a minimum, the early stopping process will continue to train for a specified number of epochs (referred to as _patience_).  If the monitored metric has not improved since the minimum by at least a threshold (_min_delta_), then training is terminated __and__ the model parameter are reset back to those that corresponded to the minimum.  Otherwise, training continues.

- ```--early_stopping``` is a switch that turns on _Early Stopping_.
- ```--early_stopping_monitor``` specifies a string that defines the model performance metric that is to be monitored.  The very common choice is _val_loss_, referring to the loss with respect to the validation data set (the default is to monitor the training set loss: _loss_).
- ```--early_stopping_min_delta``` specifies the floating point minimum required change in the monitor variable (default: 0.01).
- ```--early_stopping_patience``` specifies the integer number of epochs to continue training after the minimum has been found (default: 20).

## Text Outputs
Zero2Neuro can generate text output as it is executing.  This can be useful for monitoring the progress of the training process, as well as for debugging problems.

### Verbosity
The __verbosity level__ is an integer that controls the amount of textual output that is generated for typical users of Zero2Neuro.  By default, verbosity level is zero; this can be changed in one of several ways using the input arguments:
- ```--verbose``` or ```-v```: add one to the current verbosity level.  Each can be specified multiple times to set verbosity to higher levels.  Note that ```-v``` has a single dash, and not two.
- ```-vvv``` is shorthand for adding 3 to the current verbosity level.  This is a typical choice since both a text summary of the model architecture and the per-epoch state of the training set are printed.

### Debugging
The __debugging level__ is largely intended for use by Zero2Neuro developers.  By default, this level is zero.
- ```--debug``` and ```-d```: add one to the debugging level.
- ```-dddd``` is shorthand for adding 4 to the current debugging level.  This generates __a lot__ of debugging output during training.

## Flow Control
Not all invocations of the Zero2Neuro engine result in training of the model.  By default, if a specific experiment has already been performed, as indicated by the existence of the corresponding _*_results.pkl_ file, then Zero2Neuro will exit without loading data, creating the model, or training.

The user can control whether the model is trained using several different arguments:

- ```--nogo``` (switch) tells Zero2Neuro to load the data and create the model, but stop short of training.  This is useful for performing quick checks for correctness of the model and data.
- ```--force``` (switch) tells Zero2Neuro to perform the data loading, model creation, and training, even if the corresponding results file already exists.
- ```--data_save_folds``` specifies a string that defines the path and file name prefix to which to save the loaded data in TF-Dataset format.  One TF-Dataset file is generated per data fold.  This is useful for translating tabular data into the TF-Dataset format, which can be much more efficient for training under certain conditions (specifically, when the data set is very large).  When specified, Zero2Neuro loads the data and saves the files, but does not create the model or train it.

## Using a Saved Model for Continued Training
Instead of creating a model from scratch (and initializing its parameters randomly), it is possible to start training from a saved model in Keras format.  In these cases, the model architecture arguments are ignored.

### Loading a specific model

- ```--load_trained_model``` specifies the path to an already existing _.keras_ file.

### Checkpointing
Checkpointing is the process of occasionally saving the current model during the training process.  Should the Zero2Neuro training process be interrupted unexpectedly (power outage, or reached the allocated execution time on a supercomputer), this capability enables Zero2Neuro to recover from the last saved state and continue training.

- ```--checkpoint_model``` this switch enables checkpointing.
- ```--checkpoint_nepochs``` this specifies an integer number of epochs (default: 1) that defines how often checkpoint files should be saved in terms of training epochs.  If chosen to be too small, then the overhead of writing the files can dominate the training process.

Checkpointing behavior:
   - On start, Zero2Neuro will check for a checkpoint file.  If it finds one, then it will load the saved model and continue training from the last epoch that was saved (e.g., if the training process was interrupted after saving state at epoch 1138, then the restored model will continue training at epoch 1138).
   - Every _nepochs_, Zero2Neuro will save a new checkpoint file as long as performance has improved since the last checkpoint file (measured with respect to the _Early Stopping Monitor Variable_).
   - Zero2Neuro will attempt to delete old checkpoint files, but it may leave some.
   - The checkpoint file paths are of the form _BASE_FILE_PATH_checkpoint_DDDDD.keras_, where DDDDD is the epoch number.

Notes:
- The checkpointing process does not recover any data set or random number generator states.
- The checkpointing process is robust to interruptions during model saving.

## Processor Control
When sharing a computing node with other users, it is often important to regulate the use of the node's shared resources to prevent interfering with the work of others.  These arguments provide ways of limiting / selecting resources to use.

- ```--cpus_per_task``` specifies an integer that controls the maximum number of virtual threads used by Zero2Neuro (default: no limits).  When using a supercomputer scheduler such as SLURM, set this number to be the same as the CPUS_PER_TASK request in your batch file.  This is typically set directly in your batch file and not in a configuration file.
- ```--inter_ops``` specifies an integer that defines the degree of execution parallelism allowed for DNN computation when using the CPU to evaluate or train a model (default: 1).  The ideal choice of this argument varies widely; it is best to choose values that evenly divide _cpus_per_task_.

- ```--gpu``` switch that turns on the use of the GPU, assuming one is available.
- ```--no-gpu``` switch that turns off the use of the GPU (default state).



