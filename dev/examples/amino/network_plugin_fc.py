'''
Amino Acid Custom Network — Plugin test for Zero2Neuro.
=======================================================

This file replicates the amino/network.txt fully-connected architecture
but through the custom_network plugin system.

It demonstrates that build_model(args) can use the SAME standard flags
(--number_hidden_units, --dropout, --hidden_activation, --tokenizer, …)
as any other Zero2Neuro network type.

Amino acid input pipeline:
  raw string → TextVectorization (char tokenizer) → Embedding → Flatten → FC stack → output

Because this network uses --tokenizer + --embedding, build_model must return
a tuple: (model, text_vectorization_layer) — same contract as the built-in
fully_connected type when tokenizer=True.
Zero2Neuro's pipeline in zero2neuro.py then calls .adapt() on the
text_vectorization_layer using the training strings.

Run:
  cd zero2neuro/src
  python zero2neuro.py @../examples/amino/data.txt
                       @../examples/amino/experiment.txt
                       @../examples/amino/network_plugin.txt -vvv --force
'''

import keras
import keras.saving as saving
import keras.ops as ops


# ---------------------------------------------------------------------------
# Custom activation: elup1 = elu(x) + 1
# Keeps output strictly positive — appropriate for binding affinity (≥ 0).
# Must be registered so Keras can serialize/deserialize the model correctly.
# ---------------------------------------------------------------------------
@saving.register_keras_serializable(package='zero2neuro')
def elup1(x):
    '''elu(x) + 1 — strictly positive output activation.'''
    return ops.elu(x) + 1.0


def build_model(args):
    '''
    Build a fully-connected amino acid binding-affinity network.

    Mirrors the architecture in amino/network.txt:
      raw string → TextVectorization → Embedding → Flatten → Dense stack → output

    Parameters
    ----------
    args : argparse.Namespace
        Full Zero2Neuro configuration.  All values below come from flags
        in the .txt config file — no hard-coded defaults here.

          args.input_shape0                    (1,)  — one string per example
          args.output_shape0                   (1,)
          args.output_activation               'elup1'
          args.loss                            'mae'
          args.metrics                         ['mae', 'mse']
          args.learning_rate                   float
          args.number_hidden_units             list[int]  e.g. [200, 100, 50, 25, 10]
          args.hidden_activation               'elu'
          args.dropout                         float or None
          args.L2_regularization               float or None
          args.tokenizer_max_tokens            int    e.g. 24
          args.tokenizer_split                 'character'
          args.tokenizer_output_sequence_length int   e.g. 30
          args.embedding_dimensions            int    e.g. 15

    Returns
    -------
    tuple: (compiled keras.Model,  keras.layers.TextVectorization)
        Returns a tuple because the tokenizer layer must be adapted to the
        training data AFTER the model is built.  Zero2Neuro handles this
        automatically when it receives a tuple from build_model.
    '''

    # -----------------------------------------------------------------------
    # 1.  Output activation — 'elup1' is custom; map it to the function.
    #     Standard activations (relu, elu, linear …) pass through as strings.
    # -----------------------------------------------------------------------
    output_act = elup1 if args.output_activation == 'elup1' else args.output_activation

    # -----------------------------------------------------------------------
    # 2.  L2 regularizer — None if not set in the config file
    # -----------------------------------------------------------------------
    regularizer = (keras.regularizers.l2(args.L2_regularization)
                   if args.L2_regularization else None)

    # -----------------------------------------------------------------------
    # 3.  TextVectorization layer — converts raw strings to integer token ids.
    #     Settings all come from the standard Zero2Neuro tokenizer flags.
    # -----------------------------------------------------------------------
    text_vectorization = keras.layers.TextVectorization(
        max_tokens=args.tokenizer_max_tokens,                   # --tokenizer_max_tokens
        split=args.tokenizer_split,                             # --tokenizer_split character
        output_sequence_length=args.tokenizer_output_sequence_length,  # --tokenizer_output_sequence_length
        standardize=args.tokenizer_standardize,                 # --tokenizer_standardize
        encoding=args.tokenizer_encoding,                       # --tokenizer_encoding
    )

    # -----------------------------------------------------------------------
    # 4.  Build the model graph (Functional API)
    # -----------------------------------------------------------------------

    # Input: one raw string per example (matches --input_shape 1 in the config)
    inputs = keras.Input(shape=(args.input_shape0[0],), dtype='string', name='input')

    # Tokenizer: string → integer token ids of length tokenizer_output_sequence_length
    x = text_vectorization(inputs)

    # Embedding: integer ids → dense vectors of size embedding_dimensions
    # input_dim = vocab size (tokenizer_max_tokens), output_dim = embedding dims
    x = keras.layers.Embedding(
        input_dim=args.tokenizer_max_tokens,        # vocabulary size
        output_dim=args.embedding_dimensions,       # --embedding_dimensions
        name='embedding'
    )(x)

    # -----------------------------------------------------------------------
    # NEW: Multi-Head Self-Attention Block
    # -----------------------------------------------------------------------
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=args.embedding_dimensions,
        name='multi_head_attention'
    )(query=x, value=x, key=x)
    
    # Residual connection and layer normalization
    x = keras.layers.Add(name='attention_residual')([x, attention_output])
    x = keras.layers.LayerNormalization(name='attention_layer_norm')(x)

    # Flatten: (seq_len, embedding_dim) → (seq_len * embedding_dim,)
    # Required so Dense layers receive a flat vector
    x = keras.layers.Flatten(name='flatten')(x)

    # Dense hidden stack using the hidden-unit list from the config
    # e.g. --number_hidden_units\n200\n100\n50\n25\n10  → [200, 100, 50, 25, 10]
    for i, units in enumerate(args.number_hidden_units or []):
        x = keras.layers.Dense(
            units,
            activation=args.hidden_activation,     # --hidden_activation
            kernel_regularizer=regularizer,         # --l2
            name=f'hidden_{i}'
        )(x)
        if args.dropout:                            # --dropout (None → skip)
            x = keras.layers.Dropout(args.dropout, name=f'dropout_{i}')(x)

    # Output layer
    outputs = keras.layers.Dense(
        args.output_shape0[-1],                    # --output_shape
        activation=output_act,                     # 'elup1' → keeps output ≥ 0
        name='output'
    )(x)

    model = keras.Model(inputs, outputs, name='amino_custom_fc')

    # -----------------------------------------------------------------------
    # 5.  Compile — REQUIRED before returning
    # -----------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=args.loss,                             # --loss mae
        metrics=args.metrics or [],                 # --metrics mae mse
    )

    # -----------------------------------------------------------------------
    # 6.  Return (model, text_vectorization) tuple.
    #     Zero2Neuro calls text_vectorization.adapt(training_strings) after
    #     this function returns, then uses the adapted layer during training.
    # -----------------------------------------------------------------------
    return model, text_vectorization
