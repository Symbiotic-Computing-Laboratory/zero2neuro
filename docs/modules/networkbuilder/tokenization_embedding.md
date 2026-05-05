---
title: Tokenization and Embedding
nav_order: 40
parent: Network Builder
has_children: false
---
# Tokenization and Embedding

Tokenizaiton and embedding allow our DNN models to receive text and other discrete objects as input.

__Tokenization__ is the process of translating text into a sequence of integer values (specifically, natural numbers).  For example, each integer might represent a different word in an input sentence.   The tokenization process is implemented using a table that maps unique words to integers.  For our purposes, these sequences are a defined length (e.g., a maximum of 20 words).  Two special tokens address exceptional conditions: the __unknown token__ represents any word that is not recognized; the __nonexistent token__ is used to fill slots in the fixed length sequence that have no corresponding word.  

For example, consider the following table:

| Token | Word |
|------:|------|
| 0 | (unknown) |
| 1 | (nonexistent) |
| 2 | allow |
| 3 | and |
| 4 | as |
| 5 | discrete |
| 6 | DNN |
| 7 | embedding |
| 8 | input |
| 9 | models |
| 10 | objects |
| 11 | our |
| 12 | receive |
| 13 | text |
| 14 | to |
| 15 | tokenizaiton |

The first sentence of this page would be encoded using the following sequence of integers:

```
[15, 3, 7, 2, 11, 6, 9, 14, 12, 13, 3, 0, 5, 10, 4, 8, 1, 1, 1, 1]
```

The table can be constructed automatically from a dataset or can be declared explicitly by the user.

__Embedding__ is the process of translating a sequence of individual natural numbers into a corresponding seqeuence of feature vectors.  Internally, this is implemented a matrix that has the shape (N, M), where N is the maximum number of tokens, and M is a defined number of _embedding dimensions_.  For a token sequence of length K, this will yield a matrix that is of shape (K,M), with each row corresponding to the token at a particular position in the sequence.  Most typically, the values contained within the (N, M) embedding matrix are tuned as part of the full model training process.

Tokenization and embedding are currently supported for Fully-Connected Neural Networks, 1D Convolutional Neural Networks, and Recurrent Neural Networks.
___
## Tokenization Details
The following arguments define the behavior of the tokenizer.

1. __Turn on the tokenizer:__
```
--tokenizer
```
Notes:
- Model input shape must be (1,) (i.e., single strings)
- The data must be single strings
- This argument automatically turns on the embedding step

2. __Set the maxmimum number of tokens (N)__:
```
--tokenizer_max_tokens=N
```

3. __Standardize incoming strings__: 
```
--tokenizer_standardize=STANDARDIZE
```
where STANDARDIZE is one of:
- _lower_and_strip_punctuation_: lowercase all characters and remove punctuation
- _lower_: lowercase all characters
- _strip_punctuation_: remove punctuation

4. __String Splitting__: set the string splitting criterion
```
--tokenizer_split=SPLIT
```
where SPLIT is one of:
- whitespace (default): split at spaces, tabs, etc.
- character: each character is its own token

5. __String sequence length__: define the maximum length of a given string (in tokens) (K)
```
--tokenizer_output_sequence_length=K
```

6. __Pre-defined vocabulary__: define the tokens in the table:
```
--tokenizer_vocabulary
WORD1
WORD2
WORD3
  :
```
Notes:
- The default behavior (with no defined vocabulary) is to construct the table automatically from the training set inputs

7. __Character encoding__: define how the characters are represented internally:
```
--tokenizer_encoding=LANG
```
where LANG is:
- utf-8 (default): unicode transformation format
- latin-1 (or iso-8859-1): single byte format that combines ASCII and some Western European characters
- ascii: single byte ASCII
- utf-16: unicode transformation format (less common)
___
## Embedding Details
The following arguments define the behavior of the embedding layer.

1. Turn on embedding layer
```
--embedding
```
Notes:
- Automatically turned on when using the tokenizer
- This layer expects as input a sequence of natural numbers (integers) for each example

2. Define the dimensionality of each encoding vector:
```
--embeding_dimensions=M
```

