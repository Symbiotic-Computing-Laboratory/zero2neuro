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
        
            self.ins_training = ins
            self.outs_training = outs
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
    
