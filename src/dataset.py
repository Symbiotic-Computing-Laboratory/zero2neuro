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

class SuperDataSet:
    
    
        
    def __init__(self, args):
        self.dataset_type = None
        self.ins_training = None
        self.outs_training = None
        self.ins_validation = None
        self.outs_validation = None
        self.validation = None
        self.ins_testing = None
        self.outs_testing = None
        self.rotation = None
        self.output_mapping = None
        self.args = args

        # Convert the args to a full dataset

        if args.data_format == 'tabular':
            ins, outs, output_mapping = self.load_table(args.dataset_directory,
                                                        args.data_file,
                                                        args.data_inputs,
                                                        args.data_outputs,
                                                        args.data_output_sparse_categorical)
            # Start of rotation logic
            nfolds = 10 # Default nfolds is hardcoded to be 10
            n = len(ins) # The length of data
            train_folds = 8 # How many folds training should have
            rotation = args.rotation # get the rotation

            tr_folds, val_folds, tes_folds = SuperDataSet.calculate_nfolds(train_folds, nfolds, rotation) # Call the function to get the fold indexes for each
            # Get an array of the data being read i.e [0,1,2,3,4,5,6,7,8,....80,81]
            val_indices = SuperDataSet.calculate_indices(val_folds, nfolds, n)
            test_indices = SuperDataSet.calculate_indices(tes_folds, nfolds, n)

            # Since training indices uses 8 folds while the others use 1 fold we have to use a loop.
            train_indices = [SuperDataSet.calculate_indices(fold_i, nfolds, n) for fold_i in tr_folds]
            train_indices = np.concatenate(train_indices, axis=0, dtype=int)

            # This tells the model where to start in the data and the corrosponding true outputs.
            self.ins_training = ins[train_indices]
            self.outs_training = outs[train_indices]
            self.ins_validation = ins[val_indices]
            self.outs_validation =outs[val_indices]
            self.ins_testing =ins[test_indices]
            self.outs_testing =outs[test_indices]
            self.dataset_type = 'numpy'
            self.output_mapping = output_mapping
            
        else:
            assert False, 'Unsupported data format type (%s)'%args.data_format
        
    def describe(self):
        return {'dataset_type': self.dataset_type,
                'rotation': self.rotation,
                'output_mapping': self.output_mapping,
                }
    
    @staticmethod
    def load_table(dataset_path, file_name, input_columns, output_columns,
                   output_sparse_categorical=False):
        if dataset_path is None:
            file_path = file_name
        else:
            file_path = '%s/%s'%(dataset_path, file_name)
            
        if file_name[-3:] == 'csv':
            print("CSV file")
            df = pd.read_csv(file_path)
        
        elif file_name[-4:] == 'xlsx':
            # TODO
            print("XLSX file")

        else:
            assert False, "File type not recognized (%s)"%file_name

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

    # method takes in the amount of training folds, total number of folds, and the rotation
    @staticmethod
    def calculate_nfolds(train_folds, nfolds, rotation):
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