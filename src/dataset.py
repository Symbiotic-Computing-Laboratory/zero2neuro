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
import os
import tensorflow as tf
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
        
        #self.rotation = None
        #self.output_mapping = None
        self.args = args
        # Data table list
        self.data = []   
        self.data_groups = None

        # Fold list
        self.folds = []
        self.nfolds = 0
        self.n_train_folds = None
        self.categorical_translation = None

        # Load data
        self.load_data()
        
        # Combine some tables together
        self.generate_folds()

        # Translate tables to training/validation/testing
        self.generate_datasets()
        
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

    @staticmethod
    def is_list_of_tuples(obj):
        return isinstance(obj, list) and all(isinstance(item, tuple) for item in obj)

    @staticmethod
    def is_list_of_tf_datasets(obj):
        return isinstance(obj, list) and all(isinstance(item, tf.data.Dataset) for item in obj)
    
    def load_data(self):
        '''
        Load the full set of data files
        '''
        #####

        # Build translation table for categorical variables
        if self.args.data_columns_categorical_to_int is not None:
            # Each string is a new mapping: translate all of them
            self.categorical_translation = [SuperDataSet.parse_value_mapping(s) for s in self.args.data_columns_categorical_to_int]
                
        ####
        
        # Check list of files
        if self.args.data_files is None:
            if self.args.data_file is None:
                assert False, "No data files specified"
            else:
                # Create a list of 1
                self.args.data_files = [self.args.data_file]
        elif self.args.data_file is not None:
            assert False, "Cannot have both data_file and data_files specified"
        #####
        # Error checks

        # Must at least be a set of features for inputs
        if self.args.data_inputs is None:
            handle_error('Must specify data_inputs', self.args.debug)

        # If there are weights, then there must also be outputs
        if (self.args.data_outputs is None) and (self.args.data_weights is not None):
            handle_error("Must specify data_outputs if there are also data_weights", self.args.debug)

        ####
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
            handle_error("Each string in data_files must either be just 'file name' or a 'file name, data group'", self.args.debug)

        # Replace args.files with just the file names
        self.args.data_files = columns[0]
        
        # Second argument must be ints
        if len(columns) == 2:
            # This argument exists
            # TODO: Check to make sure data groups start at 0 and increment by 1
            # Convert to ints
            try:
                self.data_groups = [int(s) for s in columns[1]]
            except ValueError as e:
                handle_error('data_files error parsing group', self.args.debug)
            except TypeError as e:
                handle_error('data_files error parsing group', self.args.debug)

            # Check to see if any group numbers are missing
            missing = SuperDataSet.is_missing_values(self.data_groups)
            if missing is not None:
                raise ValueError(f'data_files missing groups: {missing}')


        #######
        # We are assuming that all files are the same format.  

        # Different handler for each file format type
        if self.args.data_format == 'tabular':
            self.data = self.load_table_set()
            
        elif self.args.data_format == 'tabular_indirect':
            self.data = self.load_table_indirect_set()
            
        elif self.args.data_format == 'pickle':
            self.data = self.load_pickle_set()
            
        elif self.args.data_format == 'tf-dataset':
            self.data = self.load_tf_set()
            
        else:
            assert False, "Data format %s not recognized"%self.args.data_format

        ######
        # Clean up data if composed of numpy arrays: we want each item
        #  in the list to be a 4-tuple (anything less, we will add Nones
        #  to make them 4-tuples)
        if SuperDataSet.is_list_of_tuples(self.data):
            # We will assume that each tuple is the same size
            tuple_size = len(self.data[0])
            if tuple_size < 4:
                # Add some Nones
                #print(f"ADDING tuple elements: {4-tuple_size}")
                self.data = [d + (None,) * (4-tuple_size) for d in self.data]
                
        elif SuperDataSet.is_list_of_tf_datasets(self.data):
            pass
        else:
            handle_error('Data must all be numpy arrays or all tf.data.Datasets', self.args.debug)
            
        
        #######
        print_debug(1, self.args.debug, "TOTAL DATA FILES: %d"%len(self.data))

    def generate_folds(self):
        '''
        Generate folds from the input data tables

        self.data -> self.folds
        
        '''
        
        if self.args.data_fold_split == 'identity':
            # No translation
            self.folds = self.data
        elif self.args.data_fold_split == 'group-by-file':
            if self.data_groups is not None:
                # File-level groups are defined
                print_debug(3, self.args.debug,
                            'Data groups: ' + str(self.data_groups))
                if self.args.data_representation == 'numpy':
                    self.folds = self.generate_folds_by_group_numpy()
                
                else:
                    # TODO: LUKE use sample_from_dataset()
                    self.folds = self.generate_folds_by_group_tf()

            else:
                    handle_error("No file-level groups defined", self.args.debug)
                
                
        elif self.args.data_fold_split == 'group-by-example':
            if self.args.data_representation == 'numpy':
                # TODO: Andy
                handle_error("data_fold_split group-by-example not yet supported", self.args.debug)            
                
            else:
                handle_error("data_fold_split group-by-example not supported for tf-dataset", self.args.debug)
                    

        elif self.args.data_fold_split == 'random':
            if self.args.data_representation == 'numpy':
                self.generate_folds_random_numpy()
                
            else:
                handle_error("data_fold_split random not supported for tf-dataset", self.args.debug)

        
        elif self.args.data_fold_split == 'random-stratify':
            if self.args.data_representation == 'numpy':
                # TODO: LUKE.  Need to think about whether we should just stratify based on the output or an arbitrary column
                handle_error("data_fold_split random-stratify not yet supported", self.args.debug)
                
            else:
                handle_error("data_fold_split random-stratify not supported for tf-dataset", self.args.debug)

        else:
            handle_error("data_fold_split %s not recognized."%self.args.data_fold_split, self.args.debug)
            
        self.nfolds = len(self.folds)
        print_debug(1, self.args.debug, "TOTAL DATA FOLDS: %d"%len(self.folds))

    def generate_folds_by_group_numpy(self):
        '''
        Folds by file group for numpy case
        '''
        # Numpy array case
        data_out = []
                
        # Number of pieces of information for each file (ins,) vs (ins,outs) vs (int,outs,weights) vs (ins,outs,weights,groups)
        data_size = len(self.data[0])
        print('DATA:', self.data)
        # Loop over every grouping: 0 ... K-1
        ngroups = max(self.data_groups)+1
        print_debug(2, self.args.debug, "Number of fold groups: %d"%ngroups)
        for grp in range(ngroups):
            # Accumulate all of the elements into a new list (which will become a tuple)
            data_in_group = []
            
            # Loop over every element in each data tuple (ins, outs, weights)
            for i in range(data_size):
                # Grab the numpy arrays for this element and every matching group
                # Connect the rest of the data with the group number
                data_and_group = zip(self.data, self.data_groups)
                datas = [d[i] for d, g in data_and_group if g == grp]
                # Concatenate these together along the rows
                data_in_group.append(np.concatenate(datas, axis=0))
                        
            # Add this data group to the growing list
            data_out.append(tuple(zip(data_in_group)))
            
        return data_out

    def generate_folds_by_group_tf(self):
        '''
        Folds by file group for tf datasets
        '''
        # Array for the tf datasets
        group_ds = []

        # Grab the number of folds. 
        nfolds = max(self.data_groups)+1
        print_debug(2, self.args.debug, "Number of fold groups: %d"%nfolds)

        # Loop over the data and datagroups, store the datasets for a specific group in datas
        for grp in range(nfolds):
            data_and_group = zip(self.data, self.data_groups)
            datas = [d for d, g in data_and_group if g == grp]

            
            # Make multiple tf datasets one big dataset and append that fold to group_ds
            # If only one dataset append that one dataset
            if len(datas) > 1:
                data_folded = tf.data.Dataset.sample_from_datasets(datas)
                group_ds.append(data_folded)
            else:
                group_ds.append(datas[0])
            
        return group_ds

    def generate_datasets(self):
        '''
        Translate a list of folds into a training, validation, and testing data set

        self.folds -> self.training, validation, testing
        '''
        
        if((self.folds is None) or (len(self.folds) == 0)):
            handle_error("No folds specified", self.args.debug)
            
        if self.args.data_representation == "numpy":
            # Numpy representation
            if self.args.data_set_type == "fixed":
                self.dataset_split_fixed()
            elif self.args.data_set_type == "holistic-cross-validation":
                self.split_cross_validation()
            elif self.args.data_set_type == "hold-out-cross-validation":
                self.split_cross_validation()
            elif self.args.data_set_type == "orthogonalized-cross-validation":
                # TODO
                handle_error("Dataset type not yet supported (%s)."%self.args.self_data_set_type, self.args.debug)
            else:
                handle_error("Dataset type not recognized (%s)."%self.args.self_data_set_type, self.args.debug)

            # TODO: clean this up (should not be debug, it should be verbosity)
            print_debug(4, self.args.debug, "Training Ins:" + str(self.ins_training))
            print_debug(4, self.args.debug, "Training Outs:" + str(self.outs_training))
            print_debug(4, self.args.debug, "Validation Ins:" + str(self.ins_validation))
            print_debug(4, self.args.debug, "Validation Outs:" + str(self.outs_validation))
            print_debug(4, self.args.debug, "Testing Ins:" + str(self.ins_testing))
            print_debug(4, self.args.debug, "Testing Outs:" + str(self.outs_testing))

            # Create self.validation for model.fit
            self.validation = (self.ins_validation, self.outs_validation) if self.ins_validation is not None else None

        # TF-Datasets
        elif self.args.data_representation == "tf-dataset":
            # TF Cache and repeating 
            if self.args.cache is not None:
                if self.args.cache == '':
                    self.folds = [ds.cache() for ds in self.folds]
                elif self.args.cache != '':
                    # TODO: Add error checking to make sure cache directory exists
                    self.folds = [ds.cache(os.path.join(self.args.cache, 'fold_{i}')) for i, ds in enumerate(self.folds)]
            if self.args.repeat:
                self.folds = [ds.repeat() for ds in self.folds]
                    
            if self.args.data_set_type == "fixed":
                self.tf_split_fixed()
            elif self.args.data_set_type == "holistic-cross-validation":
                # TODO: LUKE
                self.tf_split_cross_validation()
            elif self.args.data_set_type == "hold-out-cross-validation":
                # TODO: LUKE
                self.tf_split_cross_validation()
            elif self.args.data_set_type == "orthogonalized-cross-validation":
                # TODO
                handle_error('TF Datasets does not yet support orthogonalized-cross-validation.', self.args.debug)
                
            else:
                handle_error('data_set_type not recognized (%s)'%self.data_set_type, self.args.debug)
            
            #####
            # Handle final pipeline elements of training/validation/testing data sets
            if self.args.shuffle is not None:
                self.training = self.training.shuffle(self.args.shuffle)

            self.training = self.training.batch(self.args.batch, num_parallel_calls=tf.data.AUTOTUNE)

            if self.args.prefetch is None:
                self.training = self.training.prefetch(tf.data.AUTOTUNE)
            elif self.args.prefetch > 0:
                self.training = self.training.prefetch(self.args.prefetch)

            #####
            if self.validation is not None:
                self.validation = self.validation.batch(self.args.batch, num_parallel_calls=tf.data.AUTOTUNE)

                if self.args.prefetch is None:
                    self.validation = self.validation.prefetch(tf.data.AUTOTUNE)
                elif self.args.prefetch > 0:
                    self.validation = self.validation.prefetch(self.args.prefetch)
            #####
            if self.testing is not None:
                self.testing = self.testing.batch(self.args.batch, num_parallel_calls=tf.data.AUTOTUNE)

                if self.args.prefetch is None:
                    self.testing = self.testing.prefetch(tf.data.AUTOTUNE)
                elif self.args.prefetch > 0:
                    self.testing = self.testing.prefetch(self.args.prefetch)
            
        else:
            handle_error("Unrecognized data_representation (%s)"%self.args.data_representation, self.args.debug)


    def dataset_split_fixed(self):
        '''
        Assign one data group to each of training, validation, testing
        '''
        data_len = len(self.folds)
        
        if data_len > 3:
            handle_error("Cannot exceed 3 data folds for data_set_type=fixed (we have %d)"%(len(self.folds)), self.args.debug)

        # Training set
        self.ins_training = self.folds[0][0]

        if len(self.data[0]) >= 2:
            self.outs_training = self.folds[0][1]

        if len(self.data[0]) >= 3:
            self.weights_training = self.folds[0][2]
            
        if data_len >= 2:
            # Validation set is fold #1
            self.validation=self.folds[1]
            self.ins_validation = self.validation[0]
            
            if len(self.validation) >= 2:
                self.outs_validation = self.validation[1]
                
            if len(self.validation) >= 3:
                self.weights_validation = self.validation[2]

        if data_len == 3:
            # Testing set
            self.ins_testing = self.folds[2][0]

            if len(self.folds[2]) >= 2:
                self.outs_testing = self.folds[2][1]
                
            if len(self.folds[2]) >= 3:
                self.weights_testing = self.folds[2][2]
    
    def tf_split_fixed(self):
        data_len = len(self.folds)

        if data_len > 3:
            handle_error("Cannot exceed 3 data groups for split=fixed (we have %d)"%(len(self.data)), self.args.debug)

        # Training set
        self.training = self.folds[0]
            
        if data_len >= 2:
            # Validation set
            self.validation=self.folds[1]
            
        if data_len == 3:
            # Testing set
            self.testing = self.folds[2]

    def tf_split_cross_validation(self):
        pass
            

    def combine_all_data_tables(self):
        '''
        Combine all data tables into a single ins, outs, weights, groups

        With help from ChatGPT

        :return: Single tuple (ins, outs, weights, groups)
        
        '''
        
        # Transpose the self.data array of lists
        grouped = list(zip(*self.data))  
        # grouped = [(ins0, ins1, ...), (outs0, outs1, ...), ...]
        
        result = []
        
        # Iterate over the ins, outs, weights and groups
        for group in grouped:
            # Filter out None values
            arrays = [x for x in group if x is not None]

            if arrays:
                combined = np.concatenate(arrays, axis=0)
            else:
                combined = None

            result.append(combined)

        return tuple(result)


    def generate_folds_random_numpy(self):
        '''
        Translate the set of data tables into a set of folds with
        random sampling.  Specifically: self.data -> self.folds

        
        '''
        # Combine all of the data tables together into one
        ins, outs, weights, groups = self.combine_all_data_tables()

        # Number of folds
        nfolds = self.args.data_n_folds

        # Size of the dataset
        n = ins.shape[0]

        # List of indices for the examples
        arr = np.arange(n)

        # Shuffle these indices
        np.random.seed(self.args.data_seed)
        np.random.shuffle(arr)

        # Compute the indices for each fold
        # q = number within each fold, r = extras that are added to the first r folds
        q, r = divmod(n, nfolds)

        # First (n - r) chunks of size q+1, then r chunks of size q
        sizes = [q + 1] * r + [q] * (n - r)
    
        fold_inds = []
        start = 0
        for size in sizes:
            fold_inds.append(arr[start:start + size])
            start += size

        # Now slice the data
        ins_folds = [ins[inds,...] for inds in fold_inds]

        if outs is None:
            outs_folds = [None] * nfolds
        else:
            outs_folds = [outs[inds,...] for inds in fold_inds]

        if weights is None:
            weights_folds = [None] * nfolds
        else:
            weights_folds = [weights[inds,...] for inds in fold_inds]

        if groups is None:
            groups_folds = [None] * nfolds
        else:
            groups_folds = [outs[inds,...] for inds in fold_inds]

        # Create a list of tuples, one for each fold
        self.folds = list(zip(ins_folds, outs_folds, weights_folds, groups_folds))

    def split_cross_validation(self):
        '''
        self.folds -> training/validation/testing datasets
        
        For numpy arrays
        
        '''
        if self.args.data_n_folds is not None:
            if not (self.args.data_n_folds == len(self.folds)):
                handle_error("n_folds must match the number of loaded data folds", self.args.debug)
            nfolds = self.args.data_n_folds
        else:
            # Infer number of folds
            nfolds = len(self.folds)
            
        n_train_folds = self.args.data_n_training_folds # How many folds training should have

        # Default: use all available folds
        if n_train_folds is None:
            n_train_folds = nfolds - 2

        self.n_train_fodls = n_train_folds
            
        if(n_train_folds > nfolds-2):
            handle_error("n_training_folds must be <= n_folds-2", self.args.debug)

        rotation = self.args.data_rotation # get the rotation
        
        print("NFOLDS=%d; NTRAIN=%d; ROTATION=%d"%(nfolds, n_train_folds,rotation))

        # Call the function to get the fold indexes for each
        tr_folds, val_fold, test_fold = SuperDataSet.calculate_nfolds(n_train_folds,
                                                                      nfolds,
                                                                      rotation,
                                                                      self.args.data_set_type,
                                                                      self.args.debug) 
        ## Training set
        # Inputs
        self.ins_training = np.concatenate([self.folds[f][0] for f in tr_folds], axis=0)

        # Outputs
        if self.folds[tr_folds[0]][1] is not None:
            self.outs_training = np.concatenate([self.folds[f][1] for f in tr_folds], axis=0)
            
        # Weights
        if self.folds[tr_folds[0]][2] is not None:
            self.outs_training = np.concatenate([self.folds[f][2] for f in tr_folds], axis=0)
            
        ## Validation
        self.ins_validation, self.outs_validation, self.weights_validation, _ = self.folds[val_fold]

        ## Testing
        self.ins_testing, self.outs_testing, self.weights_testing, _ = self.folds[test_fold]


    def split_holistic_cross_validation_expired(self):
        # TODO: need to combine the incoming datasets first

        # TODO add sample weights and groups (?)
        ins, outs, weights, groups = self.data[0]
        
        nfolds = self.args.n_folds 
        n = len(ins) # The length of data
        n_train_folds = self.args.n_training_folds # How many folds training should have

        # Default: use all available folds
        if n_train_folds is None:
            n_train_folds = nfolds - 2
            
        if(n_train_folds > nfolds-2):
            handle_error("n_training_folds must be <= n_folds-2")
        
        rotation = self.args.rotation # get the rotation
        tr_folds, val_folds, tes_folds = SuperDataSet.calculate_nfolds(n_train_folds, nfolds, rotation, self.args.data_split) # Call the function to get the fold indexes for each
        
        # Get an array of the data being read i.e [0,1,2,3,4,5,6,7,8,....80,81]
        val_indices = SuperDataSet.calculate_indices(val_folds, nfolds, n)
        test_indices = SuperDataSet.calculate_indices(tes_folds, nfolds, n)

        # Since training indices uses 8 folds while the others use 1 fold we have to use a loop.
        train_indices = [SuperDataSet.calculate_indices(fold_i, nfolds, n) for fold_i in tr_folds]
        train_indices = np.concatenate(train_indices, axis=0, dtype=int)

        # Shuffle the data
        arr = np.arange(len(ins))
        
        # TODO: Revisit when working with seed arguments. 
        np.random.seed(self.args.data_seed)
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

    def describe(self):
        return {'dataset_type': self.dataset_type,
                'rotation': self.args.rotation,
                'nfolds': self.nfolds,
                'categorical_translation': self.categorical_translation,
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

        # Iterate over all of these data sets: each is a dict
        for d in d_all:
            # Translate numpy fields for categorical variables
            if self.categorical_translation is not None:
                # Iterate over the variable/translation table pairs 
                for var, tr_dict in self.categorical_translation:
                    if var in d.keys():
                        # var is a field in the data dict
                    
                        # Map the value in each cell to the corresponding int
                        # First convert the value in the table to a string, then do the mapping
                        # Map missing values to -999 (valid mapped values will be natural numbers)
                        map_func = np.vectorize(lambda x: tr_dict.get(str(x), -999))
                        d_tmp = map_func(d[var])
                        
                        # Detect bad values
                        d_tmp_bad = d_tmp == -999
                        
                        # Check to make sure there were not any extraneous categorical values
                        if np.any(d_tmp_bad):
                            # There are some unrecognized categorical values
                            failed_values = np.unique(d[var][d_tmp_bad])
                            handle_error(f"data_columns_categorical_to_int error: unmapped values in key '{var}': {failed_values.tolist()}",
                                         self.args.debug)

                        # All okay - copy the updated numpy array over
                        d[var] = d_tmp
            
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
        
            

    #staticmethod
    def parse_value_mapping(s):
        '''
        from ChatGPT
        '''
        key, raw_values = s.split(":", 1)
        items = [item.strip() for item in raw_values.split(",")]
        return key.strip(), {v: i for i, v in enumerate(items)}

    def load_table_set(self):
        # Right now, can only have one tabular file
        #assert len(self.args.data_files) == 1, "Only support loading single tabular files"


        ####
        data = []
        
        for f in self.args.data_files:
            
            ins, outs, weights, groups = self.load_table(self.args.dataset_directory,
                                                         self.args.data_file,
                                                         self.args.data_inputs,
                                                         self.args.data_outputs,
                                                         self.args.data_weights,
                                                         self.args.data_groups,
                                                         self.categorical_translation,
                                                         self.args.debug)
            data.append((ins, outs, weights, groups))
        
        return data



    @staticmethod
    def load_table(dataset_path:str,
                   file_name:str,
                   input_columns:[str],
                   output_columns:[str],
                   data_weights:str,
                   data_groups:str,
                   categorical_translation:list[tuple[str, dict]]=None,
                   debug_level:int=0):

        # TODO: assume that file_name is absolute path if it is needed
        if dataset_path is None:
            file_path = file_name
        else:
            # Fix path construction for any OS
            file_path = '%s/%s'%(dataset_path, file_name)
            
        df = SuperDataSet.load_tabular_file(file_path)

        ##
        # Translate dataframe columns for categorical variables
        if categorical_translation is not None:
            for col, d in categorical_translation:
                if col in df.columns:
                    # col is a column in the DataFrame
                    # Keep a copy of the original values for this col:
                    original_values = df[col].copy()
                    
                    # Map the value in each cell to the corresponding int
                    #df[col] = df[col].map(d)
                    # First convert the value in the table to a string, then do the mapping
                    df[col] = df[col].map(lambda x: d.get(str(x)))
                
                    # Check to make sure there were not any extraneous categorical values
                    unmapped = df[df[col].isna()][col].index
                    if len(unmapped) > 0:
                        # There are some unrecognized categorical values
                        failed_values = original_values.loc[unmapped]
                        handle_error(f"data_columns_categorical_to_int error: unmapped values in column '{col}': {failed_values.unique().tolist()}",
                                     debug_level)
                    
                    
        ##
        ins = None
        outs = None
        weights = None
        groups = None

        output_mapping = None
        
        if len(input_columns) > 0:
            ins = df[input_columns].values

        if len(output_columns) > 0:
            #assert len(output_columns) == 1, "Dataset only supports a single output column"
            
            #if output_sparse_categorical:
            ## Interpret the column as sparse categorical
            #categories = df[output_columns[0]].astype(pd.CategoricalDtype()).cat
            #output_mapping = dict(enumerate(categories.categories))
            #outs = categories.codes.astype(pd.SparseDtype("int", fill_value=-1)).values
                
            #else:
            # Interpret as ints or floats
            outs = df[output_columns].values

        # Some datasets will have weights associated with each example
        if data_weights is not None:
            weights = df[data_weights].values

        # Dataset groups
        if data_groups is not None:
            groups = df[data_groups].values

        return ins, outs, weights, groups

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

    def load_tf_set(self):
        # TODO: Add a path to the file names, check to make sure it exists.
        tf_datasets = []
        for datafile in self.args.data_files:
            tf_datasets.append(tf.data.Dataset.load(datafile))
        
        return tf_datasets

    # method takes in the amount of training folds, total number of folds, and the rotation
    @staticmethod
    def calculate_nfolds(n_train_folds:int, nfolds:int, rotation:int, data_split:str, debug:int=0):
        '''
        :param n_train_folds: Number of training folds
        :param nfolds: Total number of folds
        :param rotation: Cross-validation rotation
        :param data_split: Type of split (holistic-cross-validation, or hold-out-cross-validation)
        :param debug: Debug level

        :return: Tuple of [training fold list], validation fold, testing fold
        
        '''
        # TODO: Look at how rotations are handled (Should be rotations - 1 for this)
        if(data_split == 'hold-out-cross-validation'): 
            # Hold-out cross-validation
            
            # Error check for hold out
            if (rotation >= nfolds-1) or (rotation < 0):
                # Make error message and pass that along with debug level to the handle_error function.
                message = 'Rotation can be a maximum of {}; it cannot be {}'.format(nfolds - 2, rotation)
                handle_error(message, debug)

            trainfolds = ((np.arange(n_train_folds) + rotation) % (nfolds - 1))

            # Validation rotates
            valfold = (nfolds - 2 + rotation) % (nfolds - 1)

            # Fixed test
            testfold = nfolds - 1

        else:
            # Holistic cross-validation
            print("- NFOLDS=%d; NTRAIN=%d; ROTATION=%d"%(nfolds, n_train_folds,rotation))
            if (rotation >= nfolds) or (rotation < 0):
                # Make error message and pass that along with debug level to the handle_error function.
                message = 'Rotation can be a maximum of {} cannot be {}'.format(nfolds - 1, rotation)
                handle_error(message, debug)
                
            trainfolds = (np.arange(n_train_folds)+rotation) % nfolds
            valfold = (nfolds - 2 + rotation) % nfolds
            testfold = (nfolds - 1 + rotation) % nfolds

        print_debug(2, debug, "TRAINING FOLDS: " + str(trainfolds))
        print_debug(2, debug, "VALIDATION FOLD: %d"%valfold)
        print_debug(2, debug, "TESTING FOLD: %d"%testfold)
        return trainfolds, valfold, testfold

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
