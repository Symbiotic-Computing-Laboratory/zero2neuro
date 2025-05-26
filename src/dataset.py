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

        # Convert the args to a full dataset

        if args.data_format == 'tabular':
            ins, outs = self.load_table(args.dataset_directory,
                                        args.data_file,
                                        args.data_inputs,
                                        args.data_outputs)
        
            self.ins_training = ins
            self.outs_training = outs
            self.dataset_type = 'numpy'
            
        else:
            assert False, 'Unsupported data format type (%s)'%args.data_format
        

    @staticmethod
    def load_table(dataset_path, file_name, input_columns, output_columns):
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

        if len(input_columns) > 0:
            ins = df[input_columns].values
        if len(output_columns) > 0:
            outs = df[output_columns].values

        return ins, outs
    
