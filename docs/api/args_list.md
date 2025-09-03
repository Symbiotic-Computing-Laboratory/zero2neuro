## Note:  
Arguments are being added, removed, and renamed often. To get the most up to date information look in [parser.py](../../src/parser.py).  
The formatting for this arguments list is **Argument Call** | Description of argument | Expected variable type.  
If a variable type is listed as "None" that means that the argument itself is a toggle.  

# Arguments  
## Experiment Arguments  
**--experiment_name** | Sets the name of your experiment for all output files | String  
**--loss** | The name of the loss function you're wanting to use (e.g., mse, mae, categorical_crossentropy) | String  
**--metrics** | The metrics you would like to keep track of, untilized by the model itself (e.g., mse, mae) | String  
**--epochs** | The amount of epochs, how many times the model looks at the entire training dataset | Integer  
**--learning_rate** | Learning rate controls how much the model changes based on the loss error function | Float  
**--clipnorm** | TODO | Float  
    
## Results Arguments  
**--results_path** | The directory for the results files | String  
**--output_file_base** | The prefix for the output files | String  
  
## Early Stopping Arguments  
**--early_stopping** | Boolean flag for using early stopping during model training | None  
**--early_stopping_min_delta** | The minimum delta for early stopping | Float  
**--early_stopping_patience** | The patience for early stopping, how many epochs before it activates | Integer  
**--early_stopping_monitor** | Which metric to use for early stopping | String  
   
## Dataset Arguments  
**--dataset_directory** | The directory of where the data being input is | String  
**--data_format** | The format of the data (e.g, tabular, tabular-indirect, netcdf, pickle, tf-dataset) | String  
**--data_representation** | The format for the data being input (numpy, tf-dataset) | String  
**--data_split** | The method of how the data is split into training/validation/testing (e.g., fixed, by-group, random, random-stratify, holistic-cross-validation, hold-out-cross-validation, orthogonalized-cross-validation) | String  
**--data_fold_split** | Split of seperate data tables into folds (e.g., identity, group-by-file, group-by-example, random, random-stratify) | String  
**--data_set_type** | TODO | String  
**--data_n_folds** | Number of cross-validation folds total | Integer  
**--data_n_training_folds** | Number of cross-validation folds for training dataset | Integer  
**--data_rotation** | Rotation for cross-validation, depending on rotation the folds assigned to training/validation/testing will change | Integer
**--data_file** | Filename for input file | String  
**--data_files** | Used for a list of multiple filenames for input files | String  
**--data_inputs** | The columns in your data that are inputs | String  
**--data_outputs** | The columns in your data that are the outputs | String  
**--data_weights** | The column in your data that has the sample weights | String  
**--data_groups** | The column in your data that assigns the row to a corresponding dataset group | String  
**--data_output_sparse_categorical** | Translates the output column values into a sparse categorical representation | None  
**--data_columns_categorical_to_int** | Translates the output column values into a unique integer in string order (line format: VAR NAME:STR0,STR1,STR2...') | String  
**--data_columns_categorical_to_int_direct** | Translates the output column values to a specified integer (line format: VAR NAME: STR0->INT0,STR1->INT1,STR2->INT2,...') | String  
**--data_seed** | A seed for random shuffling of data into folds | Integer  
  
## TF Dataset Arguments  
**--batch** | Training set batch size, usually follows the pattern of 2^x (e.g, 2, 4, 8, 16, 32...) | Integer  
**--prefetch** | Number of batches to prefetch, the data needed will be loaded before it is needed so there's little to no downtime between training and grabbing data | Integer  
**--num_parallel_calls** | Number of threads used during batch construction | Integer  
**--cache** | Used for storing data that needs to be accessed while training in a high-speed and easily accesible location. (default: none; RAM: specify empty string; else specify file) | String  
**--shuffle** | The size of the shuffle buffer | Integer  
**--repeat** | Boolean flag for continually repeating the training set | None  
**--steps_per_epoch** | Number of training batches per epoch, must use --repeat if using this | Integer  
**--steps_per_validation_epoch** | Number of validation batches per epoch, must use (TODO) --repeat_validation if using this | Integer  
  
## High-level Commands  
**--nogo** | Sets up the data, constructs network, but does not run the experiment | None  
**--force** | Performs the experiment even if it was performed and completed previously, will overwrite previous results | None  
**--verbose** | Verbosity level of output | Expects a count, ex (-v is the lowest -vvv would be the highest)  
**--debug** | Debugging level of output, similair to verbosity but also affects how detailed the error messages are | Same as verbosity, expects -d or -ddd  
  
## CPU/GPU Arguments  
**--cpus_per_task** | Number of threads to consume during experiment | Integer  
**--gpu** | USE gpu | None  
**--no-gpu** | Do NOT use gpu | None  
  
## Post-run Arguments  
**--render_model** | Create an image of the model architecture, gets stored in results directory | None  
**--save_model** | Saves the trained model in a keras file | None  
**--no-save_model** | Do not save the trained model | None  
**--log_training_set** | Log the **FULL** training set in results pickle file | None  
**--log_validation_set** | Log the **FULL** validation set in results pickle file | None  
**--log_test_set** | Log the **FULL** testing set in results pickle file | None  
  
## Weights and Biases Arguments  
**--wandb** | Report metric data to Weights and Biases | None  
**--wandb_project** | WandB project name | String  
**--wandb_name** | WandB experiment name | String  
**--note** | Used for giving text to WandB for additional information | String  
  
## Network Specification Arguments  
**--load_trained_model** | Loads an already trained model from a previous experiment | String of trained model's filename  
**--network_type** | Type of network to create, (fully_connected, cnn) | String  
**--network-test** | Builds the network and nothing else | None  
**--input_shape0, --input_shape** | Shape of the network's input layer | List of Integers  
**--hidden_activation** | Activation function for the hidden layers (e.g., elu, relu, linear) | String  
**--number_hidden_units** | Number of hidden units per layer (a sequence of ints) | List of Integers  
**--output_shape0, --output_shape** | Shape of the network's output layer | List of Integers  
**--output_activation** | Activation function for the output layer (e.g., sigmoid, linear, softmax) | String  
**--batch_normalization** | Toggle batch normalization. | None  
  
## CNN Network Arguments  
**--conv_kernel_size** | Convolution filter size per layer (sequence of ints) | List of Integers  
**--conv_number_filters** | Convolution filters (base, output) | List of Integers  
**--conv_pool_size** | Max pooling size (0=None) | Integer  
**--conv_padding** | Padding type for convolutional layers (valid, same) | String  
**--conv_activation** | Activation function for convolutional layers (e.g., elu, relu, softmax) | String  
**--conv_batch_normalization** | Toggle for batch normalization | None  
**--conv_strides** | Strides for each convolutional layer | Integer  
  
## Regularization Parameters  
**--dropout** | Dropout rate | Float  
**--dropout_input** | Dropout rate for inputs | Float  
**--spatial_dropout** | Dropout rate for the convolutional layers | Float  
**--L1_regularization, --l1** | TODO  | Float  
**--L2_regularization, --l2** | TODO  | Float  