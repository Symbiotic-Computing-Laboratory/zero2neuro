'''
Template: User Network File for Zero2Neuro custom_network plugin
================================================================

Copy this file, rename it (e.g. my_cnn_lstm.py), and fill in build_model().

CONTRACT
--------
• Define one top-level function:  def build_model(args)
• Call model.compile() before returning.
• Return the compiled keras.Model.

HOW HYPER-PARAMETERS WORK
--------------------------
All hyper-parameters come from the standard --flag value entries in your .txt
config file — exactly like any other Zero2Neuro network type.

In your .txt file:
    --learning_rate=0.001
    --dropout=0.2
    --number_hidden_units
    128
    64
    --hidden_activation=relu

In build_model(args):
    args.learning_rate        → 0.001
    args.dropout              → 0.2
    args.number_hidden_units  → [128, 64]
    args.hidden_activation    → 'relu'

No defaults, no getattr tricks — every value comes from the config file.

WIRE INTO ZERO2NEURO
---------------------
One .txt file (same style as amino/network_rnn.txt):

    --network_type
    plugin

    --input_shape
    13

    --output_shape
    1

    --output_activation=linear
    --learning_rate=0.001
    --dropout=0.2

    --number_hidden_units
    128
    64

    --hidden_activation=relu

    --plugin_list
    custom_network-custom_network_builder
    network_file=/abs/path/to/my_custom_network.py

Run:
    python zero2neuro.py @experiment.txt @data.txt @my_network.txt
'''

import keras


def build_model(args):
    '''
    Build and return a compiled keras.Model.

    All parameters come from args — the full Zero2Neuro argparse.Namespace.
    They were set in the .txt config file using the standard --flag value format.

    Parameters
    ----------
    args : argparse.Namespace
        Full Zero2Neuro configuration. Key attributes:
          args.input_shape0          input shape, e.g. [13] or [30, 13]
          args.output_shape0         output shape, e.g. [1]
          args.output_activation     e.g. 'linear', 'sigmoid'
          args.loss                  e.g. 'mse', 'mae'
          args.metrics               list of metric name strings
          args.learning_rate         float
          args.dropout               float or None
          args.number_hidden_units   list of ints, or None
          args.hidden_activation     e.g. 'relu', 'elu'
    '''

    # --- input / output shapes ------------------------------------------------
    input_shape  = args.input_shape0
    output_units = args.output_shape0[-1]   # last dimension = number of output neurons
    output_act   = args.output_activation

    # --- build the network (Functional API — replace with your own design) ----
    inputs = keras.Input(shape=input_shape, name='input')
    x = inputs

    # Stack Dense layers using the hidden-unit list from the config file.
    # e.g. --number_hidden_units\n128\n64  → args.number_hidden_units = [128, 64]
    hidden_units = args.number_hidden_units or []
    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(units, activation=args.hidden_activation,
                               name=f'hidden_{i}')(x)
        if args.dropout:                          # None or 0.0 → skip
            x = keras.layers.Dropout(args.dropout, name=f'dropout_{i}')(x)

    outputs = keras.layers.Dense(output_units, activation=output_act,
                                 name='output')(x)
    model = keras.Model(inputs, outputs, name='my_custom_network')

    # --- compile — REQUIRED before returning ----------------------------------
    # Map the Zero2Neuro optimizer name to a Keras optimizer instance.
    # The learning rate comes from --learning_rate in the config file.
    _optimizers = {
        'adam':    keras.optimizers.Adam(learning_rate=args.learning_rate),
        'sgd':     keras.optimizers.SGD(learning_rate=args.learning_rate),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=args.learning_rate),
        'adamw':   keras.optimizers.AdamW(learning_rate=args.learning_rate),
    }
    optimizer_name = (args.optimizer or 'adam').lower()
    optimizer = _optimizers.get(optimizer_name,
                                keras.optimizers.Adam(learning_rate=args.learning_rate))

    model.compile(
        optimizer=optimizer,
        loss=args.loss,
        metrics=args.metrics or [],
    )

    return model
