---
title: Sentiment Analysis
nav_order: 60
has_children: false
parent: Zero2Neuro Examples
---

# Example: Analyzing Tweets to Determine Implied Sentiment

The dataset used for this example is made up of 1.6 million tweets (subsampled down to 30k for PC convenience). The goal of this example is to perform natural language processing on a tweet and determine the sentiment behind the tweet, meaning whether it's a positive or negative comment.
  
This form of natural language processing is used often in social media algorithms to categorize posts. For instance, companies may use a sentiment analysis model that focuses on posts about their brand in order to asess their reputation based off how their consumer describes them. 

## Data  
- [Source](dataset_source.md)  
- The dataset is a 30k example dataset [sentiment_dataset](sample_sentiment_data.csv).  
- The CSV structure is:
   1. The tweet (column: sentence)
   2. Whether the sentiment is positive (1) or negative (0) (column: sentiment)

## Data Configuration
- [Data Configuration File](demo_data.txt)  
  

## Prediction with Natural Language  
  
Natural language data counts as temporal data as it has a sequential and ordered nature. Words can change their meaning based on surrounding words (e.g., adjectives) and it is important to use models that can capture this relationship between sequence and values.

## Networks  

Provided are two different network architectures, Simple RNN and GRU. LSTM is left out but is a viable alternative. An example of an LSTM configuration can be found in the [Amino Example](../amino/README.md). 

- Because our input is made of strings, the network architecture must include two stages of preprocessing:
    1. Tokenization: Translating from type string to a token integer (different words are encoding using unique integers)
        - Since the dataset is sentences we split up by whitespace to tokenize each word.
        - We have to define a maximum number of unique tokens (words), since we are doing natural language processing this has to be pretty high. For this examples we have it set to 5,000 but feel free to experiment with different values.
        - This translation is performed prior to training
    2. Embedding: Translate the integers into vectors that capture each tokens' meaning.  This translation is learned during the training process.
- [Simple Recurrent Neural Network](demo_network_rnn.txt)
- [GRU Recurrent Neural Network](demo_network_gru.txt)


## Executing an Experiment

Basic invocation:
```
python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network.txt -vvv --force
```

where network.txt can be substituted for any of the above network configuration files.
