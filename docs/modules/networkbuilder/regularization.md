[Base Index](../../index.md)  
[Previous Index](index.md)

## Regularization  
  
Regularization techniques are used in machine learning to combat over fitting. They try to penalize the model for becoming too complicated and help move the model to being more simple and generalized to the problem. 

###  Dropout
Dropout is a regularization technique that "drops out" a fraction of the neurons in a network by setting their output to 0. This combats neurons becoming too dependent on each other and encourages the model to look at more features. Dropout uses a probability to "randomly" select neurons to disable during training. This technique is only used during training as it is best practice to have the full model active when you are going through a testing set. 

### L1 Regularization 

### L2 Regularization

### Batch Normalization