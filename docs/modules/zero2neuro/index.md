---
title: Zero2Neuro Engine
nav_order: 30
parent: Zero2Neuro Modules
has_children: true
---
  
# Zero2Neuro Engine

The Zero2Neuro engine is responsible for model training and evaluation.  Specifically, the flow of the engine execution is:

1. Parsing command line arguments and performing compatibility checks
2. Loading and organizing the data
3. Constructing and training the model
4. Evaluating model performance
5. Generating performance reports in both pickle and xlsx formats
6. Saving the model to a file for later use


Typically, the arguments associated with the engine are specified in their own _experiment.txt_ or _experiment_config.txt_ file.  The following is an example derived from [the Amino Acid Example](../../../examples/amino/README.md)
```
# Experiment and output file base names
--experiment_name=amino_acid
--results_path=./results
--output_file_base={args.experiment_name}_{args.network_type}_R{args.data_rotation:02d}

# Save the trained model
--save_model

# Generate a picture of the specified model
--render_model

# Loss function determines what is minimized during training
# MAE = Mean Absolute Error
--loss=mae

# Additional metrics that may be of interesting
# MSE = Mean Squared Error
--metrics
mae
mse

### Training details
# Size of individual training passes
--learning_rate=0.0001
# Maximum number of training passes
--epochs=5000
# How many examples to present at a time
#  (often determined by total number of examples and memory constraints)
--batch=4096

# Conditions for halting the training before the maximum number of passes is reached
#  We are looking for a low point in the monitor variable (typically an error)
# Turn on early stopping
--early_stopping
# Metric to be considered for identifying the low error point.  val_loss is 
#  often the right choice
--early_stopping_monitor=val_loss
# How many training passes to perform after the apparent minimum is reached
--early_stopping_patience=250

# Which data to include in the output pickle file
--log_training_set
--log_validation_set
--log_testing_set
```

## More Details (Intermediate)

- [Training the Model](training_model.md)
- [Saving Results](saved_results.md)  
- [Monitoring Training with Weights and Biases](wandb.md)
