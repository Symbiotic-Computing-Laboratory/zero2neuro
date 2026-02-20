# Zero2Neuro Change Log

Use inverse order

## V0.4.1 2026-02-20 Luke Sewell  
* Updated network builder documentation  

## V0.3.0 2026-02-05 Andrew H. Fagg
* Added support for LSTM and GRU networks 
* Added examples in examles/amino

## V0.2.0 2026-02-05 Andrew H. Fagg
* Support for Tokenization and Embedding in all network schemas
* Support for Recurrent Neural Networks
* Argument parser (for files) now:
   * Allows comments (everything right of '#' is ignored
   * Allows empty lines in argument files
   * Strips trailing whitespace from arguments
* Added example/amino.  Demonstrates:
   * tokenizer, embedding 
   * Solving the same problem with RNNs, CNNs, and fully connected networks
   * Regression or classification

## V0.1.7 2026-02-04 Luke Sewell
* Added ability to make results path if specified one doesn't exist.

## V0.1.6 2026-01-22 Andrew H. Fagg
* Added --tabular_header_names to provide a list of column header names for cases where there is no header row in a CSV file
* Added dataset preprocessing stage after creation of training/validation/testing set
   * If the inputs are actually strings, numpy says that they are objects.  This is corrected
* Added support for tokenization and embedding for fully connected and CNN schemas
* Added 'amino' example

## V0.1.5 2026-01-09 Andrew H. Fagg
* Added variable number of validation folds
'''
--data_n_validation_folds	
'''	
	
## 2025-12-23 Andrew H. Fagg 
* Added support for example-wise splitting of data into groups
'''
--data_fold_split=group-by-example
'''

## 2025-12-19 Luke Sewell Version 0.1.4
* Added autofitting to the xlsx reports

## 2025-12-12 Luke Sewell Version 0.1.3  
* Added error handling for xlsx reports (Maximum rows)

## 2025-12-03 Andrew H. Fagg Version 0.1.2
* Added support for categorical-to-continuous feature mapping (--data_columns_categorical_to_float_direct)
* Added elup1 non-linearity

## 2025-11-07 Luke Sewell  
* Added support for tf-datasets to use pickle file reporting functionality.  

## 2025-10-24 Luke Sewell  
* Xlsx reports now have all sheets added (Key args, full args, performance report, train/val/test reports). (Only non-tf-datasets right now)

## 2025-10-13 Luke Sewell  
* Added an arguments list sheet to xlsx export.  

## 2025-10-08 Andrew H. Fagg
* Added support for xlsx files

## 2025-09-18 Andrew H. Fagg
* Added row and column offsets for tabular data (csv only right now)

## 2025-08-20 Andrew H. Fagg
* Changed use of print_debug() to match argument order of handle_error()
* New standard: use args.debug for print_debug() and args.verbose for handle_error()
* Added --load_trained_model, allowing the user to specify a model file instead of starting from scratch

## 2025-07-22 Andrew H. Fagg
SuperDataSet:
* All numpy folds and data sets are now represented as 4-tuples.  Nones are used if a tuple entry has no data.

## 2025-07-17 Andrew H. Fagg

SuperDataSet:
* Added args.data_columns_categorical_to_int_direct

## 2025-07-06 Andrew H. Fagg

SuperDataSet:
* Data files are loaded into self.data (refer to these as *data tables*)
* generate_folds() translates self.data into self.folds.  The specific translation is defined by args.data_fold_split
* generate_datasets() translates self.folds into the training/validation/testing data sets.  The specific translation is defined by args.data_set_type
* args.data_columns_categorical_to_int is now used to declare the translation from a categorical variable to a natural number.  This mapping is defined by the user and not automatically.
* Now set up to support sample weights (but we are not using them yet).

Command Line Arguments:
* Many arguments names have been changed to reflect the module that they refer to
* Use of expired arguments now generates errors

