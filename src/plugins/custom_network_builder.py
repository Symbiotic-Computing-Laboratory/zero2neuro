'''
Custom Network Bridge Plugin for Zero2Neuro.

PHILOSOPHY: This plugin is a bridge, not a builder.
It loads the user's own .py file, calls build_model(args), and returns the
compiled model to the training pipeline.  Zero2Neuro handles everything else
(data, fit, eval, saving).

CONTRACT — user's .py must define:

    def build_model(args):
        """
        args : full Zero2Neuro argparse.Namespace
               All network hyper-parameters come from the same .txt config file
               using the standard --flag value format, e.g.:
                 args.input_shape0          input shape  (list)
                 args.output_shape0         output shape (list)
                 args.output_activation     e.g. 'linear', 'sigmoid'
                 args.loss                  e.g. 'mse', 'mae'
                 args.metrics               list of metric names
                 args.learning_rate         float
                 args.dropout               float or None
                 args.number_hidden_units   list of ints or None
                 args.hidden_activation     e.g. 'relu', 'elu'
        """
        ...
        model.compile(...)   # required
        return model         # must be a compiled keras.Model

USAGE — one network config file (same style as amino/network_rnn.txt):

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

    --plugin_list
    custom_network-custom_network_builder
    network_file=/path/to/my_network.py
'''

import os
import sys
import importlib.util
import traceback

if importlib.util.find_spec('keras') is not None:
    import keras as _keras
    _KERAS_MODEL_BASE = _keras.Model
elif importlib.util.find_spec('tensorflow.keras') is not None:
    import tensorflow.keras as _keras
    _KERAS_MODEL_BASE = _keras.Model
else:
    _KERAS_MODEL_BASE = None   # fall back to duck-typing in _validate_model

from plugin_base import GenericPlugin, require
from zero2neuro_debug import handle_error


class custom_network_builder(GenericPlugin):
    '''
    Bring-Your-Own-Network bridge plugin.

    Reads network_file from the --plugin_list line, dynamically imports it,
    and calls build_model(args) where args is the full Zero2Neuro namespace.

    The user puts ALL hyper-parameters (learning_rate, dropout, hidden units, …)
    in the same .txt network config file using the standard --flag value format.
    No comma-separated key=value on the plugin_list line.
    '''

    def __init__(self):
        super().__init__()
        self.role = 'custom_network_builder'
        self.parser.description = (
            'Bridge plugin: imports a user .py file and calls build_model(args).'
        )

        # The only plugin-level argument is the path to the user's .py file.
        # ALL network hyper-parameters go in the .txt config file as standard flags.
        self.parser.add_argument(
            '--network_file', type=str, default='my_network.py',
            help='Path to the user .py file defining build_model(args). Defaults to my_network.py.'
        )
        self.parser.add_argument(
            '--skip_compile_check', action='store_true',
            help='Skip the compile check (useful for advanced custom training loops).'
        )

        self.example = (
            '--plugin_list\n'
            'custom_network-custom_network_builder\n'
            'network_file=/path/to/my_network.py'
        )

    # -------------------------------------------------------------------------
    # Entry point — called by PluginManager
    # -------------------------------------------------------------------------

    def call(self, **kwargs):
        '''
        1. Validate network_file.
        2. Import the user module.
        3. Call build_model(args).
        4. Validate the returned model (or the model part of a tuple).
        5. Return {'model': <result>}.

        build_model may return either:
          (a) a compiled keras.Model           — simple case
          (b) (model, text_vectorization_layer) — when --tokenizer is used;
              Zero2Neuro calls .adapt() on the text_vectorization_layer after
              receiving the tuple, exactly as the built-in network types do.
        '''
        args   = require(kwargs, 'args')              # Zero2Neuro namespace
        self._validate_network_file(args)             # check file exists + is .py
        module = self._import_user_module(args)       # dynamic import
        result = self._call_build_model(module, args) # call user function

        # Unpack the result — could be a bare model or a (model, layer) tuple
        if isinstance(result, tuple):
            # Tokenizer case: (model, text_vectorization_layer)
            model = result[0]
        else:
            model = result

        self._validate_model(model, args)             # check compiled keras.Model
        return {'model': result}                      # pass result (tuple or model) upstream

    # -------------------------------------------------------------------------
    # Step 1 — validate network_file: Checks if user's file (path) exists and ends with .py
    # -------------------------------------------------------------------------

    def _validate_network_file(self, args):
        '''Raise a clear error if network_file is missing, absent on disk, or not .py.'''
        path = self.args.network_file



        path = os.path.abspath(path)

        if not os.path.exists(path):
            handle_error(
                f'custom_network: network_file not found:\n'
                f'  {path}\n'
                f'Check the path and make sure the file exists.',
                args.verbose
            )

        if not path.endswith('.py'):
            handle_error(
                f'custom_network: network_file must be a .py file, got:\n  {path}',
                args.verbose
            )

    # -------------------------------------------------------------------------
    # Step 2 — dynamically import the user's module
    # -------------------------------------------------------------------------

    def _import_user_module(self, args):
        '''
        Load the user's .py from any location on disk using importlib.
        Temporarily adds the file's directory to sys.path so sibling imports work.
        Verifies that a top-level function build_model(args) exists in the file.
        '''
        path     = os.path.abspath(self.args.network_file)
        name     = os.path.splitext(os.path.basename(path))[0]  # module name = filename stem
        user_dir = os.path.dirname(path)

        class _TemporaryPathManager:
            # 1. Setup: Check if the folder is already in the system path
            def __init__(self, dir_path):
                self.dir = dir_path
                self.added = dir_path not in sys.path
            
            # 2. Start: Temporarily add the folder to the system path
            def __enter__(self):
                if self.added:
                    sys.path.insert(0, self.dir)
            
            # 3. Cleanup: Safely remove the folder when finished (even if it crashes)
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.added and self.dir in sys.path:
                    sys.path.remove(self.dir)

        # Use the context manager to safely handle the sys.path modification
       
        with _TemporaryPathManager(user_dir):
            spec   = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module      # register so relative imports work
            spec.loader.exec_module(module)

        # The file must expose exactly one callable: build_model
        if not hasattr(module, 'build_model'):
            handle_error(
                f'custom_network: network_file must define:\n'
                f'\n'
                f'    def build_model(args): ...\n'
                f'\n'
                f'  File: {path}\n'
                f'  Public names found: {[n for n in dir(module) if not n.startswith("_")]}',
                args.verbose
            )

        return module

    # -------------------------------------------------------------------------
    # Step 3 — call build_model
    # -------------------------------------------------------------------------

    def _call_build_model(self, module, args):
        '''
        Call module.build_model(args).

        args = Zero2Neuro argparse.Namespace — contains all the network
               hyper-parameters specified in the user's .txt config file
               (learning_rate, dropout, number_hidden_units, …).
        '''
        # 1. Hit "Play": Hand the configuration (args) to the user's function and run it.
        # 2. Bubble Up: If the user wrote buggy code, we let the natural Python error 
        #    bubble up directly to the terminal instead of catching it manually.
        return module.build_model(args)

    # -------------------------------------------------------------------------
    # Step 4 — validate the returned model
    # -------------------------------------------------------------------------

    def _validate_model(self, model, args):
        '''
        Check that build_model returned a compiled keras.Model.

        Accepts:
          • A compiled keras.Model (Functional API, Sequential, subclassed)
          • The model part already extracted from a (model, text_vectorization) tuple
        Uses isinstance() when keras is available; falls back to duck-typing.
        '''
        if model is None:
            handle_error(
                'custom_network: build_model() returned None.\n'
                'Return a compiled keras.Model  (or a (model, text_vectorization) tuple).',
                args.verbose
            )

        # Type check — accept any keras.Model subclass (Sequential, Functional, subclassed)
        if _KERAS_MODEL_BASE is not None:
            if not isinstance(model, _KERAS_MODEL_BASE):
                handle_error(
                    f'custom_network: build_model() must return a keras.Model '
                    f'(or a (model, text_vectorization) tuple), got {type(model).__name__}.',
                    args.verbose
                )
        else:
            # Keras not importable at plugin load time — duck-type check
            if not all(hasattr(model, a) for a in ('fit', 'predict', 'compile')):
                handle_error(
                    'custom_network: returned object does not look like a keras.Model '
                    '(missing fit / predict / compile).',
                    args.verbose
                )

        # Compilation check (Optional) — model.optimizer is None on uncompiled models
        # We intentionally allow users to skip this compilation check.
        # If the user is writing a highly advanced custom training loop 
        # (like Diffusion or GANs), they might manage the optimizer manually.
        # By passing --skip_compile_check, they can bypass this safety net.
        # If they bypass it and forget to compile a standard model, Keras 
        # will naturally throw an error for them when training begins.
        if not getattr(self.args, 'skip_compile_check', False):
            compiled = (getattr(model, 'optimizer', None) is not None or
                        getattr(model, '_is_compiled', False))
            if not compiled:
                handle_error(
                    'custom_network: build_model() returned an uncompiled model.\n'
                    'Call model.compile(...) before returning.\n'
                    '(Use --skip_compile_check on the plugin line to bypass this for custom training loops)',
                    args.verbose
                )
