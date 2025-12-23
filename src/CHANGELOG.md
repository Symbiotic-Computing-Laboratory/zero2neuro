# Zero2Neuro Change Log

Use inverse order
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

