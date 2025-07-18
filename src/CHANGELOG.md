# Zero2Neuro Change Log

Use inverse order

##2025-07-17 Andrew H. Fagg

SuperDataSet:
* Added args.data_columns_categorical_to_int_direct

##2025-07-06 Andrew H. Fagg

SuperDataSet:
* Data files are loaded into self.data (refer to these as *data tables*)
* generate_folds() translates self.data into self.folds.  The specific translation is defined by args.data_fold_split
* generate_datasets() translates self.folds into the training/validation/testing data sets.  The specific translation is defined by args.data_set_type
* args.data_columns_categorical_to_int is now used to declare the translation from a categorical variable to a natural number.  This mapping is defined by the user and not automatically.
* Now set up to support sample weights (but we are not using them yet).

Command Line Arguments:
* Many arguments names have been changed to reflect the module that they refer to
* Use of expired arguments now generates errors

