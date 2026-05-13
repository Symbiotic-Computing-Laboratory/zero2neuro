---
title: Saving Results
nav_order: 20
parent: Zero2Neuro Engine
has_children: false
---
  
# Saving Results
Following the training of the DNN model, Zero2Neuro will evaluate the model with respect to each of the training, validation, and testing data sets.  The trained model and the evaluation results can then be saved for future use.

## Base File Path
The base file path for saving results is specified using a combination of three arguments:
- ```--results_path``` is a string that specifies the directory that the results will be written to.  The default is the local directory ("./").
- ```--experiment_name``` is a string that describes the high-level experiment that you are performing.  The default is "experiment"
- ```--output_file_base``` is a string that describes the specific experiment.  This string is evaluated at run-time and can include any command-line argument.  The best practice is that the file name be unique over all experiments that you might perform with a model.  

Example:
```
--results_path=./results
--experiment_name=dog_classifier
--output_file_base={args.experiment_name}_R{args.data_rotation:02d}
```

For N-Fold cross-validation rotation 5, the file name base will be:
```
./results/dog_classifier_R05
```

Including the rotation in the file name base is key for N-Fold cross-validation because we will be performing N different experiments (corresponding to rotations 0 ...  N-1).



## Model Keras File
The trained model can be saved as a Keras model file.  This saved model can later be reloaded and used by custom python code.

Turn on model saving:
```
--save_model
```

Turn off model saving (default):
```
--no_save_model
```

If both arguments are specified, then the last one will determine whether the model is saved or not.

The model will be saved to the file:
```
BASE_FILE_PATH_model.keras
```

## Model Diagram
An image of the specified model can be generated using the following argument:
```
--render_model
```

The image will be written to: 
```
BASE_FILE_PATH_model_plot.png
```

Example: [Model Render](../../../images/zero2neuro_render_example.png)  

## Python Pickle File
The pickle file contains detailed information about the training and evaluation process.  This file contains a single object, a dictionary, that includes the following keys:

- __fname_base__: the unique BASE_FILE_PATH for this experiment.
- __args__: a copy of the ArgumentParser arguments.
- __dataset__: a string description of the dataset.
- __history__: training history returned by model.fit().  This dictionary includes the following keys:
   - __loss__: Training loss as a function of epoch
   - __val_loss__: Validation loss as a function of epoch (if there is a validation data set)
   - For each metric M:
      - __M__: metric value for the training data set as a function of epoch
      - __val_M__: metric value for the validation data set as a function of epoch (if there is a validation data set)
- __training_loss__: Final loss for the training data set
- __validation_loss__: Final loss for the validation data set (if there is a validation data set)
- __testing_loss__: Final loss for the testing data set (if there is a testing data set)
- For each metric M:
   - __training_M__: Metric value for the training data set
   - __validation_M__: Metric value for the validation data set (if it exists)
   - __testing_M__: Metric value for the testing data set (if it exists)
- If ```--log_training_set``` is specified:
   - __ins_training__: The full set of input examples in the training set (numpy tensor)
   - __outs_training__: The full set of desired output values in the training set (numpy tensor)
   - __predict_training__: The full set of model outputs for the training set (numpy tensor)
- If ```--log_validation_set``` is specified:
   - __ins_validation__: The input examples for the validation data set
   - __outs_validation__: The desired outputs for the validation data set
   - __predict_validation__: The model outputs for the validation data set
- If ```--log_testing_set``` is specified:
   - __ins_testing__: The input examples for the testing data set
   - __outs_testing__: The desired outputs for the testing data set
   - __predict_testing__: The model outputs for the testing data set

__NOTES__: 
-  Do not log the data sets if they are large!
- The pickle file name has the form: 
```
BASE_FILE_PATH_results.pkl
```


## XLSX (Excel) Spreadsheet
Zero2Neuro supports the generation of a XLSX spreadsheet that contains the details of the experiment, including the experimental conditions and evaluation results.

The following argument will turn on the generation of the spreadsheet
```
--report
```

The generated spreadsheet will have the name:
```
BASE_FILE_PATH_report.xlsx
```

This spreadsheet will include the following _sheets_:
- __Key Arguments List__: a list of a subset of the arguments and their values.  One argument per column.
- __Performance Report__: 
   - One column for loss for each data set (training, validation, and testing).
   - One column per metric for each data set (training, validation, and testing).
- __Argument List__: exhaustive list of all arguments and their values.  One argument per column.

- If ```--report_training``` is specified, then an additional sheet is included, named __Training Data__.  This sheet includes one row per example with columns for:
   - The desired outputs and the corresponding model outputs.
   - If ```--report_training_ins``` is also specified, then columns are included for each of the input values.

- Likewise for ```--report_validation``` and ```--report_validation_ins```
- Likewise for ```--report_testing``` and ```--report_testing_ins```

__Notes:__
- The reports for the training, validation, and testing data sets will be truncated if the maximum number of allowable rows is reached (XLSX file format restriction)
- Only 1-dimensional data (inputs, desired outputs, and predictions) can be reported
