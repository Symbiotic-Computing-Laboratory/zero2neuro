'''
Normalization Plugin for Zero2Neuro
===================================
Applies a native Keras preprocessing layer to normalize input features.

Key Design Principles:
- **Strict Data-Leakage Prevention:** Statistical metrics (mean, variance) are 
  fitted exclusively on the training split.
- **Inference Ready:** The resulting fitted Keras layer is embedded inside the 
  saved model file (.keras). At deployment, preprocessing happens seamlessly 
  without requiring external scalers or pickled states.
- **N-Dimensional Native:** Natively supports both 2D (tabular) and 3D 
  (spatiotemporal) data without manual reshaping.

Usage Example (@data.txt):
    --plugin_list
    normalization-preprocess
    norm_axis=-1
'''

import keras
from plugin_base import GenericPlugin, require


class normalization(GenericPlugin):
    '''
    Normalizes input feature arrays. Returns the modified numpy arrays along
    with the fitted keras.layers.Normalization instance.
    '''

    def __init__(self):
        '''Initialize the plugin and hook it into the 'preprocess' stage.'''
        super().__init__()

        self.role = 'preprocess'
        self.parser.description = 'Native Keras input normalization plugin.'

        self.parser.add_argument(
            '--norm_axis', type=int, default=-1,
            help='Axis for Keras Normalization. -1 = per-feature (default), None = global scalar.'
        )

    def call(self, **kwargs) -> dict:
        '''
        Executes the normalization transform.

        The plugin reads the inputs from the payload, computes the normalisation 
        metrics purely from the training set, and applies the transform across 
        all available data splits.
        '''
        # Extract the required and optional data splits
        ins_train = require(kwargs, 'ins_train')
        ins_val   = kwargs.get('ins_val', None)
        ins_test  = kwargs.get('ins_test', None)

        axis = getattr(self.args, 'norm_axis', -1)

        # 1. Fit the normalization layer on training data ONLY
        norm_layer = keras.layers.Normalization(
            axis=axis,
            name='input_normalization'
        )
        norm_layer.adapt(ins_train)

        # 2. Transform the splits using the pre-fitted layer
        #    (We use .numpy() to extract the eager tensor back to a numpy array)
        results = {
            'ins_train': norm_layer(ins_train).numpy(),
            'ins_val':   norm_layer(ins_val).numpy() if ins_val is not None else None,
            'ins_test':  norm_layer(ins_test).numpy() if ins_test is not None else None,
            'normalization_layer': norm_layer
        }

        return results
