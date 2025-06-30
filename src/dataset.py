'''
Uniform handling of data sets for full Deep Learning experiments.

Multiple file formats:
- Flat representation in tabular files (csv, xlsx, xarray)
- Indirect representations (tabular files that refer to a list of image files)
- TF Datasets

Partitioning data sets for training and evaluation:
- Training set only
- Training + validation set
- Training + validation + test set
- N-fold cross-validation


'''


import pandas as pd
import numpy as np
import re
import pickle
from zero2neuro_debug import *

from PIL import Image

class SuperDataSet:
    
    
        
    def __init__(self, args):
        self.dataset_type = None
        self.ins_training = None
        self.outs_training = None
        self.weights_training = None

        self.ins_validation = None
        self.outs_validation = None
        self.weights_validation = None
        self.validation = None
        
        self.ins_testing = None
        self.outs_testing = None
        self.weights_testing = None
        self.testing = None
        
        self.rotation = None
        self.output_mapping = None
        self.args = args
        self.data = []
        self.data_groups = None

        # Load data
        self.load_data()
        
        # Combine some tables together
        self.merge_data()

        # Translate tables to training/validation/testing
        self.create_datasets()
        
    @staticmethod
    def is_missing_values(lst):
        '''
        Check to see if any values are missing between 0 and the max of a list

        From: ChatGPT
        '''
        expected = set(range(0, max(lst) + 1))
        actual = set(lst)
        missing = expected - actual
        
        return sorted(missing) if missing else None


    
    def load_data(self):
        '''
        Load the full set of data files
        '''
        # Check list of files
        if self.args.data_files is None:
            if self.args.data_file is None:
                assert False, "No data files specified"
            else:
                # Create a list of 1
                self.args.data_files = [self.args.data_file]
        elif self.args.data_file is not None:
            assert False, "Cannot have both data_file and data_files specified"

        # Must at least be a set of features for inputs
        if self.args.data_inputs is None:
            raise ValueError('Must specify data_inputs')

        # If there are weights, then there must also be outputs
        if (self.args.data_outputs is None) and (self.args.data_weights is not None):
            raise ValueError('Must specify data_outputs if there are also data_weights')


        # Individual file strings could have: just file_name or file_name, data_group
        # Parse these out
        split_rows = [re.split(r'[,\s]+', s.strip()) for s in self.args.data_files]

        # Get the set of lengths
        lengths = {len(row) for row in split_rows}
        
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent number of arguments in data_files: {sorted(lengths)}")

        # Gather by index in each list
        columns = list(map(list, zip(*split_rows)))
        if len(columns) > 2:
            raise ValueError("Each string in data_files must either be just 'file name' or a 'file name, data group'")

        # Replace args.files with just the file names
        self.args.data_files = columns[0]
        
        # Second argument must be ints
        if len(columns) == 2:
            # This argument exists

            # Convert to ints
            try:
                self.data_groups = [int(s) for s in columns[1]]
            except ValueError as e:
                raise ValueError('data_files error parsing group')
            except TypeError as e:
                raise ValueError('data_files error parsing group')

            # Check to see if any group numbers are missing
            missing = SuperDataSet.is_missing_values(self.data_groups)
            if missing is not None:
                raise ValueError(f'data_files missing groups: {missing}')


        #######

            
        # We are assuming that all files are the same format.  Could change this (but a lot more complicated)

        # Different handler for each file format type
        if self.args.data_format == 'tabular':
            self.data = self.load_table_set()
            
        elif self.args.data_format == 'tabular_indirect':
            self.data = self.load_table_indirect_set()
            
        elif self.args.data_format == 'pickle':
            self.data = self.load_pickle_set()
            
        elif self.args.data_format == 'tf_dataset':
            self.data = self.load_tf_set()
            
        else:
            assert False, "Data format %s not recognized"%self.args.data_format

        #######
        print_debug(1, self.args.debug, "TOTAL DATA FILES: %d"%len(self.data))

    def merge_data(self):
        '''
        Merge data files together by defined group
        '''
        if self.data_groups is not None:
            # Groups are defined
            if self.args.data_representation == 'numpy':
                # Numpy array case
                data_out = []
                
                # Number of pieces of information for each file (ins,) vs (ins,outs) vs (int,outs,weights)
                data_size = len(self.data[0])

                # Loop over every grouping: 0 ... K-1
                for grp in range(max(self.data_groups)+1):
                    # Accumulate all of the elements into a new list (which will become a tuple)
                    data_in_group = []
                    
                    # Loop over every element in each data tuple
                    for i in range(data_size):
                        # Grab the numpy arrays for this element and every matching group
                        # Connect the rest of the data with the group number
                        data_and_group = zip(self.data, self.data_groups)
                        datas = [d[i] for d, g in data_and_group if g == grp]
                        # Concatenate these together along the rows
                        data_in_group.append(np.concatenate(datas, axis=0))
                        
                    # Add this data group to the growing list
                    data_out.append(tuple(zip(data_in_group)))
                self.data = data_out
                
            elif self.args.data_representation == 'tf-dataset':
                raise ValueError('data_representation tf-dataset not supported')
            else:
                raise ValueError('data_representation not recognized (%s)'%self.args.data_representation)
            
        print_debug(1, self.args.debug, "TOTAL DATA GROUPS: %d"%len(self.data))
                

    def create_datasets(self):
        '''
        Translate a list of data sets into a training, validation, and testing data set
        '''
        if((self.data is None) or (len(self.data) == 0)):
            raise ValueError("No data specified")
        
        if self.args.data_representation == "numpy":
            if self.args.data_split == "fixed":
                self.split_fixed()
            elif self.args.data_split == "by-group":
                raise ValueError("Split type not supported.")
            elif self.args.data_split == "random":
                raise ValueError("Split type not supported.")
            elif self.args.data_split == "random-stratify":
                raise ValueError("Split type not supported.")
            elif self.args.data_split == "holistic-cross-validation":
                self.split_holistic_cross_validation()
            elif self.args.data_split == "hold-out-cross-validation":
                raise ValueError("Split type not supported.")
            elif self.args.data_split == "orthogonalized-cross-validation":
                raise ValueError("Split type not supported.")
            else:
                raise ValueError("Split type not recognized.")
        elif self.args.data_representation == "tf-dataset":
            raise ValueError("TF-Dataset splitting not supported.")
        else:
            raise ValueError("Unrecognized data_representation (%s)"%self.args.data_representation)

        print_debug(2, self.args.debug, "Training Ins:" + str(self.ins_training))
        print_debug(2, self.args.debug, "Training Outs:" + str(self.outs_training))
        print_debug(2, self.args.debug, "Validation Ins:" + str(self.ins_validation))
        print_debug(2, self.args.debug, "Validation Outs:" + str(self.outs_validation))
        print_debug(2, self.args.debug, "Testing Ins:" + str(self.ins_testing))
        print_debug(2, self.args.debug, "Testing Outs:" + str(self.outs_testing))


    def split_fixed(self):
        '''
        Assign one data group to each of training, validation, testing
        '''
        data_len = len(self.data)
        
        if data_len > 3:
            raise ValueError("Cannot exceed 3 data groups for split=fixed (we have %d)"%(len(self.data)))

        # Training set
        self.ins_training = self.data[0][0]

        if len(self.data[0]) >= 2:
            self.outs_training = self.data[0][1]

        if len(self.data[0]) >= 3:
            self.weights_training = self.data[0][2]
            
        if data_len >= 2:
            # Validation set
            self.validation=self.data[1]
            self.ins_validation = self.validation[0]
            if len(self.validation) >= 2:
                self.outs_validation = self.validation[1]
            if len(self.validation) >= 3:
                self.weights_validation = self.validation[2]
            

        if data_len == 3:
            # Testing set
            self.ins_testing = self.data[2][0]

            if len(self.data[2]) >= 2:
                self.outs_testing = self.data[2][1]
                
            if len(self.data[2]) >= 3:
                self.weights_testing = self.data[2][2]
        
        
    def split_holistic_cross_validation(self):
        # TODO: need to combine the incoming datasets first

        # TODO add sample weights and groups (?)
        ins, outs = self.data[0]
        
        nfolds = self.args.n_folds 
        n = len(ins) # The length of data
        n_train_folds = self.args.n_training_folds # How many folds training should have

        # Default: use all available folds
        if n_train_folds is None:
            n_train_folds = nfolds - 2
            
        if(n_train_folds > nfolds-2):
            raise ValueError("n_training_folds must be <= n_folds-2")
        
        rotation = self.args.rotation # get the rotation

        tr_folds, val_folds, tes_folds = SuperDataSet.calculate_nfolds(n_train_folds, nfolds, rotation, args.data_split) # Call the function to get the fold indexes for each
        
        # Get an array of the data being read i.e [0,1,2,3,4,5,6,7,8,....80,81]
        val_indices = SuperDataSet.calculate_indices(val_folds, nfolds, n)
        test_indices = SuperDataSet.calculate_indices(tes_folds, nfolds, n)

        # Since training indices uses 8 folds while the others use 1 fold we have to use a loop.
        train_indices = [SuperDataSet.calculate_indices(fold_i, nfolds, n) for fold_i in tr_folds]
        train_indices = np.concatenate(train_indices, axis=0, dtype=int)

        # Shuffle the data
        arr = np.arange(len(ins))
        
        # TODO: Revisit when working with seed arguments. 
        np.random.seed(8)
        np.random.shuffle(arr)

        ins = ins[arr]
        outs = outs[arr]

        self.ins_training = ins[train_indices]
        self.outs_training = outs[train_indices]
        self.ins_validation = ins[val_indices]
        self.outs_validation = outs[val_indices]
        self.validation = (ins[val_indices], outs[val_indices])
        self.ins_testing =ins[test_indices]
        self.outs_testing =outs[test_indices]
        self.dataset_type = 'numpy'

        print("DATA")
        print(self.ins_training)
        print(self.ins_validation)
        print(self.ins_testing)        
        
    def describe(self):
        return {'dataset_type': self.dataset_type,
                'rotation': self.rotation,
                'output_mapping': self.output_mapping,
                }

    @staticmethod
    def load_tabular_file(file_name:str):
        '''
        Load a CSV or XLSX file
        '''
        if file_name[-3:] == 'csv':
            print("CSV file")
            df = pd.read_csv(file_name)
        
        elif file_name[-4:] == 'xlsx':
            # TODO
            print("XLSX file")

        else:
            assert False, "File type not recognized (%s)"%file_name

        return df

    @staticmethod
    def load_pickle_file(dataset_path:str, file_name:str)->dict:

        # Figure out where the file is
        if dataset_path is None:
            file_path = file_name
        else:
            file_path = '%s/%s'%(dataset_path, file_name)

        # TODO: check that file exists and that object is a dict
        with open(file_path, "rb") as fp:
            return pickle.load(fp)
        
        return None
        
    def load_pickle_set(self):
        '''
        Load a set of pickle data files and create a set of (ins, outs, weights) for each one.

        Notes:
        - Pickle file includes one dict object
        - The referenced items in the dict for inputs, outputs, and weights are numpy arrays
        - The numpy arrays have the right shapes.
        -- Among all the inputs, the shapes are identical except in the last dimension
        -- Among all the outputs, the shapes are identical except in the last dimension
        -- The number of rows in the inputs, outputs, and weights must be the same
        '''

        data = []
        
        # Load all of the pickle files
        d_all = [SuperDataSet.load_pickle_file(self.args.dataset_directory, f) for f in self.args.data_files]
        print_debug(2, self.args.debug, "Data file list: %d"%len(d_all))
        print_debug(2, self.args.debug, "Data files: " + str(self.args.data_files))

        # Iterate over all of these data sets
        for d in d_all:
            # Contatenate all of the features along the last axis
            # TODO: check the shapes of these numpy arrays
            ins = np.concatenate([d[key] for key in self.args.data_inputs], axis=-1)
            ds = (ins,)

            if self.args.data_outputs is not None:
                # TODO: check the shapes of these numpy arrays
                # Concatenate all of the features along the last axis
                outs = np.concatenate([d[key] for key in self.args.data_outputs], axis=-1)
                ds = ds + (outs,)
            
            if self.args.data_weights is not None:
                # There is only one weight feature
                weights = d[self.args.data_weights]
                ds = ds + (weights,)

            # TODO: check the shapes of the resulting tuples

            # Append this tuple to the dataset
            data.append(ds)
        
        return data
        
            
        
    def load_table_set(self):
        # Right now, can only have one tabular file
        assert len(self.args.data_files) == 1, "Only support loading single tabular files"

        ins, outs, output_mapping = self.load_table(self.args.dataset_directory,
                                                    self.args.data_file,
                                                    self.args.data_inputs,
                                                    self.args.data_outputs,
                                                    self.args.data_output_sparse_categorical)
        self.output_mapping = output_mapping
        
        return [(ins, outs)] # TODO: add sample weights and group



    @staticmethod
    def load_table(dataset_path:str,
                   file_name:str,
                   input_columns:[str],
                   output_columns:[str],
                   output_sparse_categorical:bool=False):

        # TODO: assume that file_name is absolute path if it is needed
        if dataset_path is None:
            file_path = file_name
        else:
            # Fix path construction for any OS
            file_path = '%s/%s'%(dataset_path, file_name)
            
        df = SuperDataSet.load_tabular_file(file_path)

        ins = None
        outs = None

        output_mapping = None
        
        if len(input_columns) > 0:
            ins = df[input_columns].values

        if len(output_columns) > 0:
            assert len(output_columns) == 1, "Dataset only supports a single output column"
            
            if output_sparse_categorical:
                # Interpret the column as sparse categorical
                categories = df[output_columns[0]].astype(pd.CategoricalDtype()).cat
                output_mapping = dict(enumerate(categories.categories))
                outs = categories.codes.astype(pd.SparseDtype("int", fill_value=-1)).values
                
            else:
                # Interpret as ints or floats
                outs = df[output_columns].values

        return ins, outs, output_mapping

    def load_table_indirect_set(self):
        # Right now, can only have one tabular file
        assert len(self.args.data_files) == 1, "Only support loading single tabular-indirect files"

        ins, outs, output_mapping = self.load_table_indirect_images(self.args.dataset_directory,
                                                                    self.args.data_file,
                                                                    self.args.data_inputs,
                                                                    self.args.data_outputs,
                                                                    self.args.data_output_sparse_categorical)
        self.output_mapping = output_mapping
        return [(ins, outs)] # TODO: add sample weights and group

        
    @staticmethod
    def load_table_indirect_images(dataset_path:str,
                                   file_name:str,
                                   input_columns:[str],
                                   output_columns:[str],
                                   output_sparse_categorical:bool=False):

        # Load the table
        df = SuperDataSet.load_tabular_file(file_name)
        print(df['File'][0])

        ins = None
        outs = None

        output_mapping = None
        
        if len(input_columns) > 0:
            assert len(input_columns) == 1, "Dataset only supports a single input column containing file names"
            
            ins = SuperDataSet.load_image_set_np(dataset_path, df[input_columns].values[:,0])

        if len(output_columns) > 0:
            assert len(output_columns) == 1, "Dataset only supports a single output column"
            
            if output_sparse_categorical:
                # Interpret the column as sparse categorical
                categories = df[output_columns[0]].astype(pd.CategoricalDtype()).cat
                output_mapping = dict(enumerate(categories.categories))
                outs = categories.codes.astype(pd.SparseDtype("int", fill_value=-1)).values
                
            else:
                # Interpret as ints or floats
                outs = df[output_columns].values

        return ins, outs, output_mapping



    # method takes in the amount of training folds, total number of folds, and the rotation
    @staticmethod
    def calculate_nfolds(train_folds, nfolds, rotation, data_split):
        # TODO: Look at how rotations are handled (Should be rotations - 1 for this)
        if(data_split == 'hold-out-cross-validation'):
            trainfolds = (np.arange(train_folds) + rotation % (nfolds - 1))
            valfolds = ((nfolds - 1) - 1 + rotation) % (nfolds - 1) 
            testfolds = nfolds - 1

        else:
            trainfolds = (np.arange(train_folds)+rotation) % nfolds
            valfolds = (nfolds - 2 + rotation) % nfolds
            testfolds = (nfolds - 1 + rotation) % nfolds

        return trainfolds, valfolds, testfolds

    # method takes in a fold index, number of folds, and total number of variables in data
    @staticmethod
    def calculate_indices(fold_i, nfolds, n):
        splices = n // nfolds # Calculates how much data one fold should use

        # Where to start, i.e a dataset of 10 folds with 100 values and 10 splices per fold the eigth fold would have 
        # 10 splices * 8 fold index = starting value of 80
        start_idx = splices * fold_i 
        # Continuing on that example thie would be 80 + 10, so the data for this fold would end at 90.
        end_idx_exclusive = start_idx + splices

        # If it is the last fold go ahead and set end index at the end of dataset
        if(fold_i == nfolds - 1):
            end_idx_exclusive = n
        return np.arange(start_idx, end_idx_exclusive)

    @staticmethod
    def get_normalization_divisor(dtype):
        '''
        From chatgpt
        '''
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            return 1.0  # already normalized or leave as-is
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")
    
    @staticmethod
    def load_image_np(base_dir:str, fname:str, channels:str="RGB")->np.array:

        # Construct the full path
        # TODO: generalize this implementation
        path = f'{base_dir}/{fname}'

        # Load image and convert to np
        image = Image.open(path).convert(channels)
        arr = np.array(image)

        # Normalize to 0 ... 1
        divisor = SuperDataSet.get_normalization_divisor(arr.dtype)
        arr = arr.astype(np.float32) / divisor

        
        return arr

    @staticmethod
    def load_image_set_np(base_dir:str, fnames:[str], channels:str="RGB")->np.array:
        '''
        Load a set of images.  We assume that all images are the same size

        :param base_dir: Base directory from which the file names are relative to
        :param fnames: List of file names
        :param channels: List of expected channels for the files
        :return: Numpy array, N x rows x cols x channels
        
        '''
        images = [SuperDataSet.load_image_np(base_dir, f, channels) for f in fnames]
        # TODO: validate that all images are the same size
        return np.stack(images, axis = 0)
