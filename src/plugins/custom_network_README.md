# `custom_network` Plugin — Bring Your Own Network

**Location:** `zero2neuro/src/plugins/custom_network.py`

---

## What Is It?

A Zero2Neuro plugin that acts as a **bridge to your own Keras code**.
You write the network in your own `.py` file.
Zero2Neuro handles data loading, `model.fit()`, evaluation, and saving.

---

## How Hyper-Parameters Work

All network hyper-parameters go in the **same `.txt` config file** as standard
`--flag value` entries — exactly the same style as `amino/network_rnn.txt` or
any other Zero2Neuro config file.

```
--learning_rate=0.001
--dropout=0.2
--number_hidden_units
128
64
```

Your `build_model(args)` function reads them directly from `args`:

```python
args.learning_rate        # → 0.001
args.dropout              # → 0.2
args.number_hidden_units  # → [128, 64]
```

No special syntax, no defaults to define, no extra files.

---

## Quick Start

### 1. Write your network file

```python
# my_network.py — anywhere on disk

import keras

def build_model(args):
    # All values come from args — set by standard flags in the .txt config file.

    inputs = keras.Input(shape=args.input_shape0, name='input')
    x = inputs

    for i, units in enumerate(args.number_hidden_units or []):
        x = keras.layers.Dense(units, activation=args.hidden_activation,
                               name=f'hidden_{i}')(x)
        if args.dropout:
            x = keras.layers.Dropout(args.dropout, name=f'dropout_{i}')(x)

    outputs = keras.layers.Dense(
        args.output_shape0[-1],
        activation=args.output_activation,
        name='output'
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=args.loss,
        metrics=args.metrics or [],
    )
    return model
```

### 2. One network config file (standard format)

```
# my_network.txt

--network_type
plugin

###
# Input / Output
--input_shape
13

--output_shape
1

--output_activation=linear

###
# Hidden layers  ← your hyper-parameters, standard format
--number_hidden_units
128
64

--hidden_activation=relu

###
# Regularization
--dropout=0.2
--learning_rate=0.001

###
# Plugin — only network_file goes here
--plugin_list
custom_network-custom_network_builder
network_file=/abs/path/to/my_network.py
```

### 3. Run

```
python zero2neuro.py @experiment.txt @data.txt @my_network.txt
```

---

## `build_model` Contract

| Property | Requirement |
|---|---|
| **Name** | Must be exactly `build_model` |
| **Signature** | `def build_model(args)` |
| **`args`** | Full Zero2Neuro `argparse.Namespace` — all flags from your `.txt` file |
| **Must call** | `model.compile(...)` before returning |
| **Must return** | A compiled `keras.Model` |

---

## Key `args` Attributes

| Flag in `.txt` | `args` attribute | Type |
|---|---|---|
| `--input_shape` | `args.input_shape0` | `list[int]` |
| `--output_shape` | `args.output_shape0` | `list[int]` |
| `--output_activation` | `args.output_activation` | `str` |
| `--loss` | `args.loss` | `str` |
| `--metrics` | `args.metrics` | `list[str]` |
| `--learning_rate` | `args.learning_rate` | `float` |
| `--dropout` | `args.dropout` | `float` or `None` |
| `--number_hidden_units` | `args.number_hidden_units` | `list[int]` or `None` |
| `--hidden_activation` | `args.hidden_activation` | `str` |
| `--conv_number_filters` | `args.conv_number_filters` | `list[int]` or `None` |
| `--rnn_filters` | `args.rnn_filters` | `list[int]` or `None` |

All standard Zero2Neuro flags are available in `args`.

---

## Supported Keras Flavours

All accepted: standalone `keras`, `tensorflow.keras`, Functional API,
Sequential, and subclassed `keras.Model`.

---

## Error Reference

| Error | Fix |
|---|---|
| `network_file is required` | Add `network_file=/path/to/file.py` to the `--plugin_list` line |
| `network_file not found` | Check the path; use an absolute path |
| `network_file must be a .py file` | Provide a Python source file |
| `must define build_model` | Add `def build_model(args): ...` at module level |
| `SyntaxError in network_file` | Fix the reported line in your file |
| `ImportError` | Install missing libraries used in your file |
| `build_model() returned None` | Add `return model` |
| `must return a keras.Model` | Return the result of `keras.Model(...)` or `keras.Sequential(...)` |
| `returned an uncompiled model` | Call `model.compile(...)` before `return` |

---

## Worked Examples

### Fully-Connected MLP

```
# my_mlp.txt

--network_type
plugin

--input_shape
13

--output_shape
1

--output_activation=linear

--number_hidden_units
256
128
64

--hidden_activation=relu
--dropout=0.3
--learning_rate=0.001

--plugin_list
custom_network-custom_network_builder
network_file=/path/to/my_mlp.py
```

```python
# my_mlp.py
import keras

def build_model(args):
    inputs = keras.Input(shape=args.input_shape0)
    x = inputs
    for i, units in enumerate(args.number_hidden_units):
        x = keras.layers.Dense(units, activation=args.hidden_activation,
                               name=f'fc_{i}')(x)
        if args.dropout:
            x = keras.layers.Dropout(args.dropout)(x)
    outputs = keras.layers.Dense(args.output_shape0[-1],
                                 activation=args.output_activation,
                                 name='output')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                  loss=args.loss, metrics=args.metrics or [])
    return model
```

---

### CNN-LSTM for Time-Series

```
# my_cnn_lstm.txt

--network_type
plugin

--input_shape
30
13

--output_shape
1

--output_activation=linear
--conv_number_filters
32
64

--rnn_filters_last
64

--number_hidden_units
32

--dropout=0.2
--learning_rate=0.001

--plugin_list
custom_network-custom_network_builder
network_file=/path/to/my_cnn_lstm.py
```

```python
# my_cnn_lstm.py
import keras

def build_model(args):
    # args.conv_number_filters → [32, 64]  (from --conv_number_filters)
    # args.rnn_filters_last    → 64         (from --rnn_filters_last)
    # args.number_hidden_units → [32]       (from --number_hidden_units)

    inputs = keras.Input(shape=args.input_shape0)  # (30, 13)

    x = inputs
    for i, f in enumerate(args.conv_number_filters or []):
        x = keras.layers.Conv1D(f, kernel_size=3, activation='relu',
                                padding='same', name=f'conv_{i}')(x)
    if args.dropout:
        x = keras.layers.Dropout(args.dropout)(x)

    if args.rnn_filters_last:
        x = keras.layers.LSTM(args.rnn_filters_last, name='lstm')(x)

    for i, units in enumerate(args.number_hidden_units or []):
        x = keras.layers.Dense(units, activation=args.hidden_activation,
                               name=f'dense_{i}')(x)

    outputs = keras.layers.Dense(args.output_shape0[-1],
                                 activation=args.output_activation,
                                 name='output')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                  loss=args.loss, metrics=args.metrics or [])
    return model
```

---

## Advanced Architectures (Attention, Diffusion, GANs)

The standard Zero2Neuro `network_builder.py` **only** supports traditional topologies (Fully Connected, CNN, and RNN/LSTM/GRU). It **does not** support Attention layers, Transformers, or generative Diffusion models out-of-the-box. 

However, because the `custom_network` plugin simply receives your data and runs `model.fit()` on whatever `keras.Model` you return, you can use the plugin to build **any** advanced architecture!

### Multi-Head Attention & Transformers
You can build Transformers by simply instantiating standard Keras attention layers in your `.py` file.

```python
import keras
from keras import layers

def build_model(args):
    inputs = keras.Input(shape=args.input_shape0)
    
    # Text Tokenization & Embedding (if needed)
    text_vec = layers.TextVectorization(max_tokens=args.tokenizer_max_tokens)
    x = text_vec(inputs)
    x = layers.Embedding(args.tokenizer_max_tokens, args.embedding_dimensions)(x)

    # Add Multi-Head Self Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=args.embedding_dimensions
    )(query=x, value=x, key=x)

    # Residual connection & LayerNorm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Flatten & Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(args.output_shape0[-1], activation=args.output_activation)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=args.loss)
    
    # Remember to return the tuple if using text tokenization!
    return model, text_vec
```

### Diffusion Models & Custom Training Loops
Diffusion models (and GANs/VAEs) usually require custom training loops (e.g., adding noise at specific timesteps, computing loss on noise predictions). 

You can accomplish this seamlessly by creating a **custom Keras `Model` subclass** in your file and overriding the `train_step(self, data)` method. When Zero2Neuro calls `fit()`, Keras automatically delegates to your custom logic!

```python
import keras
import tensorflow as tf

# 1. Subclass keras.Model to override the training step
class DiffusionModel(keras.Model):
    def __init__(self, network, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def train_step(self, data):
        x, y = data # Unpack the Zero2Neuro data batch
        
        # ... Implement your diffusion noise scheduling here ...
        
        with tf.GradientTape() as tape:
            noisy_x = x # (Add noise to x)
            predictions = self.network(noisy_x, training=True)
            loss = self.compute_loss(y=y, y_pred=predictions)

        # Apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

# 2. Return the custom model in the plugin builder
def build_model(args):
    inputs = keras.Input(shape=args.input_shape0)
    outputs = keras.layers.Dense(args.output_shape0[-1])(inputs)
    core_network = keras.Model(inputs, outputs)

    # Wrap the core network in your custom Diffusion handler
    diffusion_model = DiffusionModel(core_network)
    diffusion_model.compile(optimizer='adam')
    
    return diffusion_model
```

---

## Plugin Contract Summary

| Rule | Detail |
|---|---|
| File can be anywhere | Use an absolute path in `network_file=` |
| `build_model` signature | `def build_model(args)` — one argument |
| Hyper-parameters | Standard Zero2Neuro flags in the `.txt` config file |
| Model must be compiled | Call `model.compile()` inside `build_model` before `return` |
| Role string | Must be `custom_network_builder` (set in `--plugin_list` suffix) |
