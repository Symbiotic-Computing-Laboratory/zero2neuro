[Base Index](../../index.md)  
[Previous Index](index.md)  
# Recurrent Neural Network

## Introduction

A reccurent neural network (RNN) is designed for tasks that require the processing of sequential data. RNNs have the ability to process the sequential data by keeping a memory of previous inputs. RNNs are often used for time series problems, natural language processing, and any problem where either context or order matters to the result.

## Key Components

### Recurrent Layers
Reccurent layers take in the current input and also take the memory of past inputs (called a hidden state)
- Simple RNN: A basic reccurent layer that has only one hidden state
- Long Short-Term Memory (LSTM):
- Gated Recurent Unit (GRU):

### Sequence Handling
RNNs process the sequences in data step by step. This simulates the ability of memory and allows the model to "remember" previous inputs. Sequence length and batch size are very important to keep in mind when training an RNN model.

### Hidden states
The hidden state is a trainable memory that remembers information about past inputs in a sequence.
- For what are called stacked RNNs, each layer of the RNN can have its own hidden state
- The hidden state can be returned at either each timestep or at only the final time step

### Output Types
Outputs depend on the task
- Sequence to one: An RNN can produce a single output for a input sequence (ex. sentiment analysis)
- Sequence to sequence: RNNs can also produce an output at each time step which causes a sequential output (ex. language translation)

## Example RNN Configuration
```
--experiment_name=sentiment_140_rnn
--network_type=rnn
--rnn_type=simple
--input_shape
1
--rnn_reverse_time
--rnn_filters
60
--rnn_pool_average_last
2
--rnn_filters_last
40
--rnn_dropout=0.2
--number_hidden_units
20
10
5
--hidden_activation=tanh
--output_shape
1
--output_activation=sigmoid
```

### Explanation:
TODO:  
This is the network config for the sentiment140 example.

### Oddities
- RNNs can suffer from vanishing gradients when sequences are too long (model starts to forget)
- LSTM and GRU were built to address the vanishing gradient problem and are preferred for longer sequences
- Training is often slower than with a fully connected or convolutional neural network because of sequential processing. 
