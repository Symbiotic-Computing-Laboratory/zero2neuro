'''
Zero2Neuro Command Line Parser

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import argparse

def create_parser(description='Zero2Neuro'):
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description=description, fromfile_prefix_chars='@')

    # Experiment details
    parser.add_argument('--experiment_name', type=str, default='experiment', help="Prefix for all output files");

    # TODO: change to list
    parser.add_argument('--loss', type=str, default='mse', help="Loss function name")
    # TODO: add loss_weights for model.compile step

    # TODO: change to list of lists
    parser.add_argument('--metrics', nargs='+', type=str, default=[], help="Metrics to compute")
    parser.add_argument('--rotation', type=int, default=None, help='Expired')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--learning_rate', '--lrate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--clipnorm', type=float, default=None, help="Norm clipping threshold")

    # Results
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--output_file_base', type=str, default='network', help='Output file prefix')


    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', help='Use Early Stopping during training')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.01, help="Minimum delta for early termination")
    parser.add_argument('--early_stopping_patience', type=int, default=20, help="Patience for early termination")
    parser.add_argument('--early_stopping_monitor', type=str, default="loss", help="Metric to monitor for early termination")


    # Dataset details
    parser.add_argument('--dataset_directory', type=str, default=None, help='Data set location')
    parser.add_argument('--training_mode', type=str, default=None, help='EXPIRED')

    parser.add_argument('--data_format', type=str, default=None, help='Incoming format for the data (tabular, tabular-indirect, netcdf, pickle, tf-dataset)')
    parser.add_argument('--data_representation', type=str, default='numpy', help='Internal format for the data (numpy, tf-dataset)')
    # TOOD: fix set of options
    parser.add_argument('--data_split', type=str, default=None, help='Deprecated')
    parser.add_argument('--data_fold_split', type=str, default='identity', help='Split of data tables into folds (identity, group-by-file, group-by-example, random, random-stratify)')
    parser.add_argument('--data_set_type', type=str, default=None, help='Split of data into training/validation/testing sets (fixed, holistic-cross-validation, hold-out-cross-validation, orthogonalized-cross-validation)')
    
    parser.add_argument('--n_folds', type=int, default=None, help='EXPIRED.  Use data_n_folds')
    parser.add_argument('--n_training_folds', type=int, default=None, help='EXPIRED Use data_n_training_folds')
    
    parser.add_argument('--data_n_folds', type=int, default=None, help='Number of cross-validation folds')
    parser.add_argument('--data_n_training_folds', type=int, default=None, help='Number of cross-validation folds for training')
    parser.add_argument('--data_n_validation_folds', type=int, default=1, help='Number of cross-validation folds for validation (default = 1)')
    parser.add_argument('--data_rotation', type=int, default=0, help='Cross-validation rotation')
    
    parser.add_argument('--data_file', type=str, default=None, help='Input data file')
    parser.add_argument('--data_files', nargs='+', type=str, default=None, help='Input data file list')
    #parser.add_argument('--data_table_merge', nargs='+', type=str, default=None, help='Table merge specification')   # removed
    parser.add_argument('--data_inputs', nargs='+', type=str, default=None, help='Columns in the table that are inputs')
    parser.add_argument('--data_outputs', nargs='+', type=str, default=None, help='Columns in the table that are outputs')
    parser.add_argument('--data_weights', type=str, default=None, help='Column in the table that are the sample weights')
    parser.add_argument('--data_groups', '--data_folds', type=str, default=None, help='Column in the table that correspond to the dataset group')
    parser.add_argument('--data_stratify', type=str, default=None, help='Column in the table that correspond to the stratification class')  # TODO: implement

    # For tabular data, control which row/col the headers are
    parser.add_argument('--tabular_header_row', type=int, default=None, help='Row that contains the table headers (0/None = first row; the following row is the start of the data)')
    parser.add_argument('--tabular_column_range', nargs=2, type=int, default=None, help='Column range that contains the headers/data ([10,15] means use columns 10,11,12,13,14,15)')
    parser.add_argument('--tabular_column_list', nargs='+', type=int, default=None, help='List of column range that contain the headers/data')

    
    parser.add_argument('--data_output_sparse_categorical', action='store_true', help='Translate output column into sparse categorical representation')
    parser.add_argument('--data_columns_categorical_to_int', nargs='+', type=str, default=None, help='Translation of categorical variable to a unique integer in string order (line format: VAR NAME:STR0,STR1,STR2...')
    parser.add_argument('--data_columns_categorical_to_int_direct', nargs='+', type=str, default=None, help='Translation of categorical variable to a specified integer (line format: VAR NAME:STR0->INT0,STR1->INT1,STR2->INT2,...')
    parser.add_argument('--data_columns_categorical_to_float_direct', nargs='+', type=str, default=None, help='Translation of categorical variable to a specified float value for input features (line format: VAR NAME:STR0->FLOAT0,STR1->FLOAT1,STR2->FLOAT2,...')
    parser.add_argument('--data_seed', type=int, default=1138, help='Random seed used for shuffling data into folds')

    # TF Dataset configuration
    parser.add_argument('--batch', type=int, default=None, help="Training set batch size")
    parser.add_argument('--prefetch', type=int, default=None, help="Number of batches to prefetch")
    parser.add_argument('--num_parallel_calls', type=int, default=4, help="Number of threads to use during batch construction")
    parser.add_argument('--cache', type=str, default=None, help="Cache (default: none; RAM: specify empty string; else specify file")
    parser.add_argument('--shuffle', type=int, default=None, help="Size of the shuffle buffer")
    
    #parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    parser.add_argument('--repeat', action='store_true', help='Continually repeat training set')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of training batches per epoch (must use --repeat if you are using this)")
    parser.add_argument('--steps_per_validation_epoch', type=int, default=None, help="Number of validation batches per epoch (must use --repeat_validation if you are using this)")
    


    # High-level commands
    #parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Set up data, construct network, but do not perform the experiment')
    parser.add_argument('--force', action='store_true', help='Perform the experiment even if the it was completed previously')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('--debug', '-d', action='count', default=0, help="Debugging level")

    
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--inter_ops', type=int, default=1, help="Number of parallel operations to allow")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu', help='Do not use the GPU')


    # Post
    parser.add_argument('--render_model', action='store_true', default=False , help='Write model image')
    parser.add_argument('--save_model', action='store_true', default=False , help='Save a model file')
    parser.add_argument('--no-save_model', action='store_false', dest='save_model', help='Do not save a model file')
    parser.add_argument('--log_training_set', action='store_true', default=False , help='Log the full training set in results')
    parser.add_argument('--log_validation_set', action='store_true', default=False , help='Log the full validation set in results')
    parser.add_argument('--log_testing_set', action='store_true', default=False , help='Log the full test set in results')

    # Post: reporting
    parser.add_argument('--report', action='store_true', default=False, help='Generate a report file (xlsx format)')

    parser.add_argument('--report_training', action='store_true', default=False, help='Include the training data set results in the report')
    parser.add_argument('--report_training_ins', action='store_true', default=False, help='Include the training data set inputs in the report')
    
    parser.add_argument('--report_validation', action='store_true', default=False, help='Include the validation data set results in the report')
    parser.add_argument('--report_validation_ins', action='store_true', default=False, help='Include the validation data set inputs in the report')
    
    parser.add_argument('--report_testing', action='store_true', default=False, help='Include the testing data set results in the report')
    parser.add_argument('--report_testing_ins', action='store_true', default=False, help='Include the testing data set inputs in the report')
    

    # Weights and Biases (WandB)
    parser.add_argument('--wandb', action='store_true', help='Report to WandB')
    parser.add_argument('--wandb_project', type=str, default='Supernetwork', help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default='{args.experiment_name}', help='WandB experiment name')
    parser.add_argument('--note', type=str, default=None, help="Just text to give to WandB")

    # Network specification
    parser.add_argument('--load_trained_model', type=str, default=None, help='Load a trained model instead of creating a new one')
    parser.add_argument('--optimizer', type=str, default=None, help ='Optimizer used for model training (default = Adam).')
    
    parser.add_argument('--network_type', type=str, default=None, help='Type of network to create')
    #parser.add_argument('--network_test', action='store_true', help='Build the network, but nothing else')

    parser.add_argument('--input_shape0', '--input_shape', nargs='+', type=int, default=[10], help='Shape of the network input')
    parser.add_argument('--hidden_activation', type=str, default='elu', help='Activation function for hidden fully-connected layers')
    parser.add_argument('--number_hidden_units', nargs='*', type=int, default=None, help='Number of hidden units per layer (sequence of ints)')

    # TODO: parser.add_argument("--mylist", nargs='+', action='append', type=int)
    parser.add_argument('--output_shape0', '--output_shape', nargs='+', type=int, default=[10], help='Shape of the network output')

    parser.add_argument('--output_activation', type=str, default=None, help='Activation function for output layer')
    parser.add_argument('--batch_normalization', action='store_true', help='Turn on batch normalization')

    # CNN network parameters
    parser.add_argument('--conv_kernel_size', nargs='+', type=int, default=None, help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_number_filters', nargs='+', type=int, default=None, help='Convolution filters (base, output)')

    
    parser.add_argument('--conv_pool_size', nargs='+', type=int, default=None, help='Max pooling size (default=None)')
    parser.add_argument('--conv_pool_average_size', nargs='+', type=int, default=None, help='Averaging pooling size (default=None)')
    parser.add_argument('--conv_padding', type=str, default='valid', help='Padding type for convolutional layers')
    parser.add_argument('--conv_activation', type=str, default='elu', help='Activation function for convolutional layers')
    parser.add_argument('--conv_batch_normalization', action='store_true', help='Turn on batch normalization for convolutional layers')
    parser.add_argument('--conv_strides', nargs='+', type=int, default=None, help='Strides for each convolutional layer')


    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--dropout_input', type=float, default=None, help='Dropout rate for inputs')
    parser.add_argument('--spatial_dropout', '--sdropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")


    #####################
    #  WORK ON?
    #

    # Specific experiment configuration
    #parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')

    return parser

