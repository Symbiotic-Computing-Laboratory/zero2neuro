# Network Building  
  
## Fully Connected Networks  
  
If the user specifices they want a fully connect network, network builder calls a function inside of keras3 tools that creates the fully connected network. This call contains user-defined arguments or defaults such as...  
  
```
input shape  
n of hidden units  
output shape  
dropout  
activation function  
batch normalization  
learning rate  
loss  
metrics  
```
  
For more information on what these arguments do [Arguments List](../../../api/args_list.md)  
  
## Convolutional Neural Networks

Like with fully connected networks, network builder calls a function in keras3 for CNNs but with more arguments added such as...
  
```  
conv_kernel_size  
conv_padding  
conv_number_filters  
conv_activation  
conv_pool_size  
conv_strides  
spatial_dropout
conv_batch_normalization
```

For more information on what these arguments do [Arguments List](../../..api/args_list.md)