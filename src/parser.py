'''
Zero2Neuro Command Line Parser

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
'''

import argparse
import shlex
import re
from collections.abc import Callable
from typing import Any, Union

import numpy as np
from itertools import product
from zero2neuro_debug import *


class CommentedArgumentParser(argparse.ArgumentParser):
    '''
    Wrapper around ArgumentParser that deals with:
    1.  Comments
    2.  Empty lines

    From ChatGPT
    '''
    
    def convert_arg_line_to_args(self, line):
        # Remove comments, strip only the RIGHT side (your requirement)
        line = line.split("#", 1)[0].rstrip()

        # Skip blank / whitespace-only lines
        if not line.strip():
            return []

        s = line.lstrip()

        # Option lines
        if s.startswith("-"):
            opt = s.strip()

            # Support --opt=value (value may contain spaces)
            if opt.startswith("--") and "=" in opt:
                left, right = opt.split("=", 1)
                # right already has no trailing whitespace because of rstrip() above
                return [left, right]

            # Normal option line, no splitting
            return [opt]

        # Value lines: keep internal whitespace, but trailing is already stripped
        return [line]

    
#######################
def create_parser(description='Zero2Neuro'):
    '''
    Create argument parser
    '''
    # Parse the command-line arguments
    #parser = argparse.ArgumentParser(description=description, fromfile_prefix_chars='@')
    parser = CommentedArgumentParser(description=description, fromfile_prefix_chars='@')

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

    # Preprocessor: not using right now
    #parser.add_argument('--data_preprocess_tokenize', action='store_true', help='Enable the tokenizer')


    # For tabular data, control which row/col the headers are
    parser.add_argument('--tabular_header_row', type=int, default=None, help='Row that contains the table headers (0/None = first row; NONE = no header; the following row is the start of the data)')
    parser.add_argument('--tabular_header_names', nargs='+', type=str, default=None, help='List of column names (replaces a row that contains column names)')
    parser.add_argument('--tabular_column_range', nargs=2, type=int, default=None, help='Column range that contains the headers/data ([10,15] means use columns 10,11,12,13,14,15)')
    parser.add_argument('--tabular_column_list', nargs='+', type=int, default=None, help='List of column range that contain the headers/data')
    parser.add_argument('--tabular_encoding', type=str, default=None, help='Character encoding of tabular file')

    
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
    
    # Tokenizer
    parser.add_argument('--tokenizer', action='store_true', help='Enable the tokenizer')
    parser.add_argument('--tokenizer_max_tokens', type=int, default=None, help ='Tokenizer maximumn number of tokens')
    parser.add_argument('--tokenizer_standardize', type=str, default='lower_and_strip_punctuation', help ='String standardization before tokenization')
    parser.add_argument('--tokenizer_split', type=str, default='whitespace', help ='Splitting pattern between tokens')
    parser.add_argument('--tokenizer_output_sequence_length', type=int, default=None, help='Maximum number of tokens in any given input')
    parser.add_argument('--tokenizer_vocabulary', nargs='+', type=str, default=None, help='Pre-defined tokens to load into the tokenizer')
    parser.add_argument('--tokenizer_encoding', type=str, default='utf-8', help='Tokenizer character encoding')

    parser.add_argument('--embedding', action='store_true', help='Enable the embedding layer')
    parser.add_argument('--embedding_dimensions', type=int, default=None, help='Number of embedding dimensions')

    # Fully-connected networks
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
    parser.add_argument('--batch_normalization_input', action='store_true', help='Turn on batch normalization for input layers')


    # CNN network parameters
    parser.add_argument('--conv_kernel_size', nargs='+', type=int, default=None, help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('--conv_number_filters', nargs='+', type=int, default=None, help='Convolution filters (base, output)')

    
    parser.add_argument('--conv_pool_size', nargs='+', type=int, default=None, help='Max pooling size (default=None)')
    parser.add_argument('--conv_pool_average_size', nargs='+', type=int, default=None, help='Averaging pooling size (default=None)')
    parser.add_argument('--conv_padding', type=str, default='valid', help='Padding type for convolutional layers')
    parser.add_argument('--conv_activation', type=str, default='elu', help='Activation function for convolutional layers')
    parser.add_argument('--conv_batch_normalization', action='store_true', help='Turn on batch normalization for convolutional layers')
    parser.add_argument('--conv_strides', nargs='+', type=int, default=None, help='Strides for each convolutional layer')

    # RNN network parameters
    parser.add_argument('--rnn_type', type=str, default=None, help='RNN type (simple, gru, lstm)')
    parser.add_argument('--rnn_filters', nargs='*', type=int, default=None, help='Number of filters at each RNN layer (return_sequence=True)')
    parser.add_argument('--rnn_filters_last', type=int, default=None, help='Number of filters for an optional last RNN layer (return_sequence=False)')
    parser.add_argument('--rnn_activation', type=str, default='tanh', help='Activation function for RNN layers')
    parser.add_argument('--rnn_dropout', type=float, default=None, help='Dropout rate for RNN layers')
    parser.add_argument('--rnn_L1_regularization', '--rnn_l1', type=float, default=None, help="L1 regularization parameter for RNN layers")
    parser.add_argument('--rnn_L2_regularization', '--rnn_l2', type=float, default=None, help="L2 regularization parameter for RNN layers")
    parser.add_argument('--rnn_unroll', action='store_true', help='Unroll the RNN layers across time (more efficient, but requires more memory)')
    parser.add_argument('--rnn_pool_average_last', type=int, default=None, help='Average pool size before last (return_sequences=False) RNN layer')
    parser.add_argument('--rnn_reverse_time', action='store_true', help='Reverse the time dimension of the input data.')

    # Regularization parameters
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--dropout_input', type=float, default=None, help='Dropout rate for inputs')
    parser.add_argument('--spatial_dropout', '--sdropout', type=float, default=None, help='Dropout rate for convolutional layers')
    parser.add_argument('--L1_regularization', '--l1', type=float, default=None, help="L1 regularization parameter")
    parser.add_argument('--L2_regularization', '--l2', type=float, default=None, help="L2 regularization parameter")

    # Many experiments
    parser.add_argument('--cartesian_arguments', nargs='+', type=str, default=None, help='Define a range of values for arguments')
    parser.add_argument('--cartesian_selection_index', type=int, default=None, help='Determine which argument set to select')



    return parser



#####################
class CartesianExperimentControl():
    '''
    CartesianExperimentControl
    
    Author: Andrew H. Fagg
    Modified by: Alan Lee
    
    Translate a dictionary containing parameter/list pairs (key/value) into a Cartesian product
    of all combinations of possible parameter values.
    
    Internally, the Cartesian product is stored as a list of dictionaries (parameter/value pairs).  
    This class allows for indexed access to this list.  In addition the values of a particular element
    of the list can be added to the property list of an existing object.
    
    Example:
    # Dictionary of possible parameter values
    p = {'rotation': range(20),
         'Ntraining': [1,2,3,5,10,18],
         'dropout': [None, .1, .2, .5]}
    
    # Create job iterator
    ap = CartesianExperimentControl(p)
    
    # Select the ith element of the Cartesian product list.
    # Add properties to object obj; the names of these  properties
    #  are the keys from p and the values are the specific combination
    #  of values in the ith element
    ap.set_attributes_by_index(i, obj)
    
    '''
    def __init__(self, parser:CommentedArgumentParser, params:list[str]|dict[str, []], verbose=0):
        '''
        Constructor

        :param parser: the argument parser that the args came from
        :param params: Dictionary of key/list pairs OR a string that specifies these
        :param verbose: Verbosity level
        
        '''
        self.parser = parser
        
        if isinstance(params, list):
            # Input is a list of strings: translate to a dict
            try:
                params = CartesianExperimentParser.parse_var_list(params)
            except ValueError as e:
                handle_error(f'--cartesian_arguments: {e}', verbose)

        # Store this dict
        self.params = params

        self.args_dict = params
        
        # List of all combinations of parameter values
        self.product = list(dict(zip(params,x))for x in product(*params.values()) )
        
        # Iterator over the combinations 
        self.iter = (dict(zip(params,x))for x in product(*params.values()))

    def _get_action(self, name:str)->argparse.Action:
        '''
        Extract the details about the specified parser argument

        :param name: String name of the argument in question
        :return: Argparse Argument action details
        '''

        # Find the argument in the list that matches the name
        actions = [action for action in self.parser._actions if action.dest == name]
        
        if len(actions) > 0:
            # Found the argument (there will only be one item in the list)
            return actions[0]
        else:
            # Did not find the argument
            return None
        
    def get_index(self, i):
        '''
        Return the ith combination of parameters
        
        :param i: Index into the Cartesian product list
        :return: The ith combination of parameters
        '''
        
        return self.product[i]

    def get_njobs(self):
        '''
        :return: The total number of combinations of arguments
        '''
        
        return len(self.product)

    def get_index_iterator(self):
        '''
        Return an iterator over the valid indices

        :return: A new index iterator
        '''
        return range(self.get_njobs())
    
    def set_attributes_by_index(self, i, obj):
        '''
        For an arbitrary object (typical is 'args'), set the attributes
        to match the ith job parameters
        
        :param i: Index into the Cartesian product list
        :param obj: Argument Namespace object (to be modified)
        :return: A string representing the combinations of parameters
        '''
        
        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        
        # Iterate over the arguments and their new values
        for k,v in d.items():
            # Does this argument exist in the namespace?
            if not hasattr(obj, k):
                handle_error(f'--cartesian_arguments: argument **{k}** is not valid.', obj.verbose)
                
            # Check that the new value type matches the type of the argument
            action = self._get_action(k)
            if (v is None) or isinstance(v, action.type):
                # Set the new argument value
                setattr(obj, k, v)
            else:
                # Error messages for typing mismatch
                if isinstance(v, str):
                    handle_error(f'--cartesian_arguments: argument **{k}** type mismatch for value {v}; expecting {action.type}, but received {type(v)}\n\t(Hint: some argument value in the list is a string and not numeric).', obj.verbose)
                else:
                    handle_error(f'--cartesian_arguments: argument **{k}** type mismatch for value {v}; expecting {action.type}, but received {type(v)}\n\t(Hint: check that all argument values are expected type).', obj.verbose)
            
        return self.get_param_str(i)
    
    def get_param_str(self, i):
        '''
        Return the string that describes the ith job parameters.
        Useful for generating file names
        
        @param i Index into the Cartesian product list
        '''
        
        out = 'JI_'
        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        # Iterate over the parameters
        for k,v in d.items():
            out = out + "%s_%s_"%(k,v)

        # Return all but the last character
        return out[:-1]


            
#######################
class CartesianExperimentParser():
    '''
    Safely translate a list of strings into a dict.  The strings are one of the following forms:
    ARG:VAL0, VAL1, VAL2,..., VALk-1
       where VALi is None, an int, or a float

    START, END, SKIP are ints:
    ARG:range(END)                          -> 0,1,2,3,..., END-1
    ARG:range(START,END)                    -> START, START+1, START+2, ...., END-1
    ARG:range(START,END,SKIP)               -> START, START+SKIP, START+2*SKIP, ..., END-1

    START, END, SKIP are floats:
    ARG:arange(START, END, SKIP)            -> START, START+SKIP, START+2*SKIP, ..., END-epsilon
    
    START, END are floats, NUM, BASE are ints:
    ARG:logspace(START, END, NUM[, BASE=10])   -> NUM items in the range START^BASE ... END^BASE arranged exponentially
    ARG:exp_range(START, END, NUM, BASE=10])   -> NUM items in the range START ... BASE arranged exponentially


    Examples:
    range(20)                 -> [0, 1, 2, 3, ..., 19]
    range(0,20,2)             -> [0, 2, 4, 6, ..., 18]
    arange(0, .5, .1)         -> [0, .1, .2, .3, .4]
    arange(0, .51, .1)        -> [0, .1, .2, .3, .4, .5]
    logspace(-5, 0, 6)        -> [.00001, .0001, .001, .01, .1, 1.0]
    exp_range(1, 100000, 6)   -> [1, 10, 100, 1000, 10000, 100000]
    exp_range(.00001, 1.0, 6) -> [.00001, .0001, .001, .01, .1, 1.0]
    '''

    # Type hint for the return values: lists of single types or None
    VALUE_LIST = Union[list[int|None], list[float|None], list[str|None]]

    @staticmethod
    def parse_var_list(strings: list[str]) -> dict[str, VALUE_LIST]:
        """
        Parse a list of lines.
        
        
        Parses a list of strings like:
        [
            "a:1,2,3",
            "b:range(4)",
            "c:np.arange(0,1,0.25)"
        ]

        Returns:
        {
            "a": [1,2,3],
            "b": [0,1,2,3],
            "c": [0.0,0.25,0.5,0.75]
        }
        """
        result: dict[str, VALUE_LIST] = {}

        # Loop over lines
        for s in strings:
            # Parse a single line
            var, values = CartesianExperimentParser.parse_var_values(s)

            if var in result:
                raise ValueError(f"Duplicate variable name: {var!r}")

            # Add it to the return dict
            result[var] = values

        # Complete
        return result

    # String matching function calls
    _CALL_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_\.]*)\s*\(\s*(?P<args>.*)\s*\)\s*$")

    # String matching list of numeric values
    _NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    @staticmethod
    def _split_args(arg_str: str) -> list[str]:
        '''
        Split out a list of values from a string, removing surrounding whitespace

        :param str: String list of values separated by commas
        :return: List of cleaned-up values
        
        '''
        # Simple arg splitter: supports numbers like -1, 0.5, 1e-3
        # (Not a full parser for nested calls; intentional for safety.)

        # Catch the null case
        if arg_str.strip() == "":
            return []

        # Split out and clean up all of the values
        return [a.strip() for a in arg_str.split(",")]



    @staticmethod
    def _parse_numbers(tokens: list[str]) -> Union[list[int|None], list[float|None], list[str|None], None]:
        '''
        Convert the list of tokens into a set of values.

        Return is either an int list or a float list (and individual values can be None)
        '''

        # No tokens
        if not tokens:
            return []

        # Accumulated list of parsed values
        parsed = []
        # Track if we have seen a float value
        saw_float = False

        # Loop over the tokens
        for t in tokens:
            if t == "None":
                # Found None value: this is allowed
                parsed.append(None)
                continue

            if not CartesianExperimentParser._NUM_RE.match(t):
                # Not a numeric value - return error
                return None 

            if re.fullmatch(r"[+-]?\d+", t):
                # Integer
                parsed.append(int(t))
                
            else:
                # Float
                parsed.append(float(t))
                saw_float = True

        # If any float present, promote ints to float (except None)
        if saw_float:
            parsed = [
                float(x) if isinstance(x, int) else x
                for x in parsed
                ]

        # Return the list of values
        return parsed

    @staticmethod
    def _parse_raw_list(expr: str) -> VALUE_LIST:
        '''
        Split out a list of values from a string

        :param str: String containing values that are comma separated
        :return: List of values (ints, floats, or strings).  Each can include None value
        '''
        
        # Split out the individual values
        items = [x.strip() for x in expr.split(",") if x.strip() != ""]

        # Translate into numeric values
        nums = CartesianExperimentParser._parse_numbers(items)

        # Check for error
        if nums is not None:
            # List of ints, or list of floats
            return nums

        # Some non-numeric values: keep as strings
        return items 

    @staticmethod
    def _exp_range(start: float, stop: float, num: int, base: float = 10.0) -> np.ndarray:
        '''
        Exponential spacing between VALUES `start` and `stop` (inclusive), using given base.
        
        Equivalent to:
        base ** linspace(log_base(start), log_base(stop), num)

        :param start: Starting point of values
        :param stop: Stopping point of values (inclusive
        :param int: Number of values in the sequence
        :param base: Exponential base
        '''

        # Check for valid values
        if start <= 0 or stop <= 0:
            raise ValueError("exp_range requires start>0 and stop>0 (log domain).")

        # Default: return empty list
        if num <= 0:
            return np.array([], dtype=float)

        # Compute the list
        lo = np.log(start) / np.log(base)
        hi = np.log(stop) / np.log(base)
        
        return base ** np.linspace(lo, hi, int(num))


    @staticmethod
    def _eval_whitelisted_call(expr: str) -> list[Any]:
        '''
        Evaluate a function call.  We only allow specific calls for safety

        Form is ARG:FUNC(VAL0, VAL1, ... VALk-1)

        :param str: Function call of the form ARG:FUNC(VAL0, VAL1, ... VALk-1)
        :return: List of values generated by the function call
        '''
        
        # Compare string to template
        m = CartesianExperimentParser._CALL_RE.match(expr)
        if not m:
            raise ValueError(f"Looks like a call but couldn't parse: {expr!r}")

        # Argument name
        name = m.group("name")

        # List of values
        args_s = m.group("args")

        # Split the list of values and parse them
        arg_tokens = CartesianExperimentParser._split_args(args_s)
        nums = CartesianExperimentParser._parse_numbers(arg_tokens)

        # Invalid values discovered
        if nums is None:
            raise ValueError(f"Only numeric arguments are allowed in calls: {expr!r}")

        # whitelist of callable names
        whitelist: dict[str, Callable[..., Any]] = {
            "range": range,
            "arange": np.arange,
            "np.arange": np.arange,

            # exponential grids
            "logspace": np.logspace,
            "np.logspace": np.logspace,
            "exp_range": CartesianExperimentParser._exp_range,
            }

        # Did not find the specified function
        if name not in whitelist:
            raise ValueError(f"Function not allowed: {name!r}")

        # Extract the specific func
        fn = whitelist[name]

        # range requires ints
        if name == "range":
            if not all(isinstance(x, int) for x in nums):
                raise ValueError("range() arguments must be integers")

            # Call the function and return the values
            return list(fn(*nums))  # type: ignore[arg-type]

        # Enforce exp_range third arg is int-ish
        if name == "exp_range":
            # Must have 3 or 4 values
            if len(nums) not in (3, 4):
                raise ValueError("exp_range(start, stop, num[, base]) expects 3 or 4 args.")

            # Parse out the values
            start = float(nums[0])
            stop = float(nums[1])
            num = nums[2]

            # Check type of num
            if not isinstance(num, int):
                # allow "4.0" style floats from parsing
                if float(num).is_integer():
                    num = int(num)
                else:
                    raise ValueError("exp_range 'num' must be an integer.")

            # Base
            base = float(nums[3]) if len(nums) == 4 else 10.0

            # Call the function and return the values
            return list(fn(start, stop, num, base))

        # For np.logspace: require num as int-ish (numpy will coerce, but we validate)
        if name in ("logspace", "np.logspace"):
            # Must have 3 or 4 values
            if len(nums) not in (3, 4):
                raise ValueError("logspace(start_exp, stop_exp, num[, base]) expects 3 or 4 args.")

            # Num must be an int
            if not isinstance(nums[2], int) and not float(nums[2]).is_integer():
                raise ValueError("logspace 'num' must be an integer.")
            
            nums2 = list(nums)

            # Force num to be an int
            nums2[2] = int(nums2[2])  # num

            # Call the funciton and return the list
            return list(fn(*nums2))

        # For np.arange: require num as int-ish (numpy will coerce, but we validate)
        if name in ("arange", "np.arange"):
            # Must have 3 values
            if len(nums) not in (3,):
                raise ValueError("arange(start_exp, stop_exp, num) expects 3 args.")

            # Call the funciton and return the list
            return list(fn(*nums))


        # Default: others.  Should not get here
        return list(fn(*nums))

    @staticmethod
    def parse_var_values(s: str) -> tuple[str, VALUE_LIST]:
        '''
        Parse a single line of one of the forms:

        ARG: VAL0, VAL1, VAL2, ..., VALk-1
        ARG: FUNC(VAL0, VAL1, ... VALk-1)
        
        :param s: String line
        :return: Tuple of string ARG and list of values
        '''

        # Bad format
        if ":" not in s:
            raise ValueError(f"Expected 'var:expr' format, got: {s!r}")

        # Separate argument from RHS
        var, expr = s.split(":", 1)
        var = var.strip()
        expr = expr.strip()

        # Check ARG name
        if not var:
            raise ValueError(f"Missing variable name in: {s!r}")

        # Detect function that must be evaluated by presence of (...) at the top level
        if "(" in expr and expr.rstrip().endswith(")"):
            return var, CartesianExperimentParser._eval_whitelisted_call(expr)

        # Just a list of values
        return var, CartesianExperimentParser._parse_raw_list(expr)

