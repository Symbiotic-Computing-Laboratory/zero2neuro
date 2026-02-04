[Base Index](../../index.md)  
[Previous Index](index.md)

## Regularization  
  
Regularization techniques are used in machine learning to combat over fitting. They try to penalize the model for becoming too complicated and help move the model to being more simple and generalized to the problem. 

###  Dropout
Dropout is a regularization technique that "drops out" a fraction of the neurons in a network by setting their output to 0. This combats neurons becoming too dependent on each other and encourages the model to look at more features. Dropout uses a probability to "randomly" select neurons to disable during training. This technique is only used during training as it is best practice to have the full model active when you are going through a testing set. 

### L1 Regularization
L1 works by giving a penalty to the loss function that is equal to the absolute value of the sum of coefficents. This can essentially "turn off" some of the features and focus only on the most impactful ones. It is good in situations with datasets with large numbers of features, some of which may not hold much weight in the problem being solved. 

### L2 Regularization
L2 also punishes high value coefficents however it uses the squared sum instead of the absolute. By using a squared sum L2 can never fully "turn off" a feature but can only push their coeffienct **towards** zero. 

### Batch Normalization
Batch normalization 

### Early Stopping
Early stopping is a technique that helps prevent overfitting by tracking the validation set's performance. Once the validation performance is no longer improving it is a sign that the model is no longer generalizing and is prone to overfitting. Early stopping detects this as it happens and stops the training process before the damage to generalization is too severe. In Zero2Neuro early stopping is enable via a `--early_stopping` command and can be further customized with the `--early_stopping_monitor` and `--early_stopping_patience` commands. 