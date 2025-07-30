## Note:  
Arguments are being added, removed, and renamed often. To get the most up to date information look in [parser.py](../../src/parser.py).  

# Arguments  
## Experiment Arguments  
**--experiment_name** | Sets the name of your experiment for all output files.  
**--loss** | The name of the loss function you're wanting to use (e.g., mse, mae, categorical_crossentropy)  
**--metrics** | The metrics you would like to keep track of, untilized by the model itself (e.g., mse, mae)  
**--epochs** | The amount of epochs, how many times the model looks at the entire training dataset.  
**--learning_rate** | Learning rate controls how much the model changes based on the loss error function.  
**--clipnorm** | TODO  
    
## Results Arguments  
**--results_path** | The directory for the results files  
**--output_file_base** | The prefix for the output files  
  
## Early Stopping Arguments  
**--early_stopping** | Boolean flag for using early stopping during model training.  
**--early_stopping_min_delta** | The minimum delta for early stopping.  
**--early_stopping_patience** | The patience for early stopping, how many epochs before it activates.  
**--early_stopping_monitor** | Which metric to use for early stopping.  
   
## Dataset Arguments  
**--dataset_directory** | The directory of where the data being input is.  
**--data_format** | The format of the data (e.g, tabular, tabular-indirect, netcdf, pickle, tf-dataset)  
**--data_representation** | The format for the data being input (numpy, tf-dataset)  
**--data_split** | The method of how the data is split into training/validation/testing (e.g., fixed, by-group, random, random-stratify, holistic-cross-validation, hold-out-cross-validation, orthogonalized-cross-validation)  
**--data_fold_split** | Split of seperate data tables into folds (e.g., identity, group-by-file, group-by-example, random, random-stratify)  
**--data_set_type** | TODO  
**--data_n_folds** | Number of cross-validation folds total.  
**--data_n_training_folds** | Number of cross-validation folds for training dataset.  
**--data_rotation** | Rotation for cross-validation, depending on rotation the folds assigned to training/validation/testing will change.  
**--data_file** | Filename for input file.  
**--data_files** | Used for a list of multiple filenames for input files.  
**--data_inputs** | The columns in your data that are inputs.  
**--data_outputs** | The columns in your data that are the outputs.  
**--data_weights** | The column in your data that has the sample weights.  
**--data_groups** | The column in your data that assigns the row to a corresponding dataset group.  
**--data_output_sparse_categorical** | Translates the output column values into a sparse categorical representation.  
**--data_columns_categorical_to_int** | Translates the output column values into a unique integer in string order (line format: VAR NAME:STR0,STR1,STR2...')  
**--data_columns_categorical_to_int_direct** | Translates the output column values to a specified integer (line format: VAR NAME: STR0->INT0,STR1->INT1,STR2->INT2,...')  
**--data_seed** | A seed for random shuffling of data into folds.  
  
## TF Dataset Arguments  
**--batch** | Training set batch size, usually follows the pattern of 2^x (e.g, 2, 4, 8, 16, 32...)  
**--prefetch** | Number of batches to prefetch, the data needed will be loaded before it is needed so there's little to no downtime between training and grabbing data.  
**--num_parallel_calls** | Number of threads used during batch construction.  
**--cache** | Used for storing data that needs to be accessed while training in a high-speed and easily accesible location. (default: none; RAM: specify empty string; else specify file)  
**--shuffle** | The size of the shuffle buffer.  
**--repeat** | Boolean flag for continually repeating the training set.  
**--steps_per_epoch** | Number of training batches per epoch, must use --repeat if using this.  
**--steps_per_validation_epoch** | Number of validation batches per epoch, must use (TODO) --repeat_validation if using this.  
  
## High-level Commands  
**--nogo** | Sets up the data, constructs network, but does not run the experiment.  
**--force** | Performs the experiment even if it was performed and completed previously, will overwrite previous results.  
**--verbose** | Verbosity level of output.  
**--debug** | Debugging level of output, similair to verbosity but also affects how detailed the error messages are.  
  
## CPU/GPU Arguments  
**--cpus_per_task** | Number of threads to consume during experiment.  
**--gpu** | USE gpu.  
**--no-gpu** | Do NOT use gpu.  
  
## Post-run Arguments  
**--render_model** | Create an image of the model architecture, gets stored in results directory.  
**--save_model** | Saves the trained model in a keras file.  
**--no-save_model** | Do not save the trained model.  
**--log_training_set** | Log the **FULL** training set in results pickle file.  
**--log_validation_set** | Log the **FULL** validation set in results pickle file.  
**--log_test_set** | Log the **FULL** testing set in results pickle file.  
  
## Weights and Biases Arguments  
**--wandb** | Report metric data to Weights and Biases.  
**--wandb_project** | WandB project name.  
**--note** | Used for giving text to WandB for additional information.  
  
## Network Specification Arguments  
**--network_type** | Type of network to create, (fully_connected, cnn).  
**--network-test** | Builds the network and nothing else.  
**--input_shape0, --input_shape** | Shape of the network's input layer.  
**--hidden_activation** | Activation function for the hidden layers (e.g., elu, relu, linear).  
**--number_hidden_units** | Number of hidden units per layer (a sequence of ints).  
**--output_shape0, --output_shape** | Shape of the network's output layer.  
**--output_activation** | Activation function for the output layer (e.g., sigmoid, linear, softmax)  
**--batch_normalization** | Toggle batch normalization.  
  
## CNN Network Arguments  
**--conv_kernel_size** | Convolution filter size per layer (sequence of ints)  
**--conv_number_filters** | Convolution filters (base, output)  
**--conv_pool_size** | Max pooling size (0=None).  
**--conv_padding** | Padding type for convolutional layers (valid, same)  
**--conv_activation** | Activation function for convolutional layers (e.g., elu, relu, softmax)  
**--conv_batch_normalization** | Toggle for batch normalization.  
**--conv_strides** | Strides for each convolutional layer.  
  
## Regularization Parameters  
**--dropout** | Dropout rate.  
**--dropout_input** | Dropout rate for inputs.  
**--spatial_dropout** | Dropout rate for the convolutional layers.  
**--L1_regularization, --l1** | TODO  
**--L2_regularization, --l2** | TODO  