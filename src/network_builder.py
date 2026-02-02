import os
import sys

# Import keras3_tools from proper path
neuro_path = os.getenv("NEURO_REPOSITORY_PATH")
assert neuro_path is not None, "Environment variable NEURO_REPOSITORY_PATH must be set to directory above zero2neuro and keras3_tools"

sys.path.append(neuro_path + '/keras3_tools/src/')
from fully_connected_tools import *
from cnn_tools import *
from keras.models import load_model
from zero2neuro_debug import *

class NetworkBuilder:
    
    @staticmethod
    def args2model(args):
        if args.load_trained_model is not None:
            # Load an already trained model
            model = load_model(args.load_trained_model)
            print_debug('Model %s loaded'%args.load_trained_model, 1, args.debug)

        else:
            # Build a new model
            if args.network_type == 'fully_connected':
                models = FullyConnectedNetwork.create_fully_connected_network(input_shape=args.input_shape0,
                                                                             batch_normalization_input=args.batch_normalization_input,
                                                                             n_hidden=args.number_hidden_units,
                                                                             output_shape=args.output_shape0,
                                                                             dropout_input=args.dropout_input,
                                                                             name_base='fc',
                                                                             activation=args.hidden_activation,
                                                                             lambda1=args.L1_regularization,
                                                                             lambda2=args.L2_regularization,
                                                                             dropout=args.dropout,
                                                                             # name_last='output',
                                                                             activation_last=args.output_activation,
                                                                             batch_normalization=args.batch_normalization,
                                                                             learning_rate=args.learning_rate,
                                                                             loss=args.loss,
                                                                             metrics=args.metrics,
                                                                             opt=args.optimizer,
                                                                             tokenizer=args.tokenizer,
                                                                             embedding=args.embedding,
                                                                             tokenizer_max_tokens=args.tokenizer_max_tokens,
                                                                             tokenizer_standardize=args.tokenizer_standardize,
                                                                             tokenizer_split=args.tokenizer_split,
                                                                             tokenizer_output_sequence_length=args.tokenizer_output_sequence_length,
                                                                             tokenizer_vocabulary=args.tokenizer_vocabulary,
                                                                             tokenizer_encoding=args.tokenizer_encoding,
                                                                             embedding_dimensions=args.embedding_dimensions,
                                                                              )
                # Deal with variable number of returns
                if isinstance(models, tuple):
                    model, model_text_vectorization = models
                    
                else:
                    model = models
                    model_text_vectorization = None
                    
                
            elif args.network_type == 'cnn':
                models = ConvolutionalNeuralNetwork.create_cnn_network(input_shape=args.input_shape0,
                                                                      batch_normalization_input=args.batch_normalization_input,
                                                                      conv_kernel_size=args.conv_kernel_size,
                                                                      conv_padding=args.conv_padding,
                                                                      conv_number_filters=args.conv_number_filters,
                                                                      conv_activation=args.conv_activation,
                                                                      conv_pool_average_size=args.conv_pool_average_size,
                                                                      conv_pool_size=args.conv_pool_size,
                                                                      conv_strides=args.conv_strides,
                                                                      spatial_dropout=args.spatial_dropout,
                                                                      conv_batch_normalization=args.conv_batch_normalization,
                                                                      n_hidden=args.number_hidden_units,
                                                                      output_shape=args.output_shape0,
                                                                      dropout_input=args.dropout_input,
                                                                      name_base='cnn',
                                                                      activation=args.hidden_activation,
                                                                      lambda1=args.L1_regularization,
                                                                      lambda2=args.L2_regularization,
                                                                      dropout=args.dropout,
                                                                      name_last='output',
                                                                      activation_last=args.output_activation,
                                                                      batch_normalization=args.batch_normalization,
                                                                      learning_rate=args.learning_rate,
                                                                      loss=args.loss,
                                                                      metrics=args.metrics,
                                                                      opt=args.optimizer,
                                                                      tokenizer=args.tokenizer,
                                                                      embedding=args.embedding,
                                                                      tokenizer_max_tokens=args.tokenizer_max_tokens,
                                                                      tokenizer_standardize=args.tokenizer_standardize,
                                                                      tokenizer_split=args.tokenizer_split,
                                                                      tokenizer_output_sequence_length=args.tokenizer_output_sequence_length,
                                                                      tokenizer_vocabulary=args.tokenizer_vocabulary,
                                                                      tokenizer_encoding=args.tokenizer_encoding,
                                                                      embedding_dimensions=args.embedding_dimensions,
                                                                      )
                # Deal with variable number of returns
                if isinstance(models, tuple):
                    model, model_text_vectorization = models
                    
                else:
                    model = models
                    model_text_vectorization = None
                    
                                               
            else:
                handle_error('Unsupported network type (%s)'%args.network_type, args.verbose)

        if args.debug >= 4:
            NetworkBuilder.recursive_summary(model)

        if model_text_vectorization is not None:
            return model, model_text_vectorization
        else:
            return model

    @staticmethod
    def recursive_summary(layer, indent=0):
        '''
        Recursively give a detailed summary of a model

        :param layer: A model or a layer
        :param indent: Indent level
        '''
        
        pad = " " * indent
        cls_name = layer.__class__.__name__

        #######################
        # --- helpers ---------
        def shape_from_tensor(t):
            """Get a nice shape string from a Tensor or KerasTensor (or list of them)."""
            if t is None:
                return "?"
            if isinstance(t, (list, tuple)):
                return "[" + ", ".join(shape_from_tensor(x) for x in t) + "]"
            try:
                # KerasTensor / Tensor have .shape that is a tf.TensorShape
                s = t.shape
                return str(tuple(int(d) if d is not None else None for d in s))
            except Exception:
                return "?"

        def safe_attr(obj, name):
            return getattr(obj, name, None)

        def fmt_weights(weights):
            """Format a list of variables: show name + shape."""
            if not weights:
                return "[]"
            return "[" + ", ".join(
                f"{w.name}: {tuple(w.shape)}"
                for w in weights
                ) + "]"

        def dtype_from_tensor(t):
            """Get dtype string from a Tensor/KerasTensor (or list of them)."""
            if t is None:
                return "?"
            if isinstance(t, (list, tuple)):
                return "[" + ", ".join(dtype_from_tensor(x) for x in t) + "]"

            # KerasTensor / Tensor typically has .dtype
            dt = getattr(t, "dtype", None)
            if dt is None:
                return "?"

            # Assume dt is a string
            return dt
        
            # dt can be tf.DType or a string, normalize to a friendly string
            #try:
            #    return dt.name  # tf.DType -> "float32"
            #except Exception:
            #    return str(dat)

        def layer_dtype_fallback(l):
            """Best-effort dtype when no input/output tensors exist yet."""
            # Keras 3 commonly exposes compute_dtype; other attrs may exist too.
            for attr in ("compute_dtype", "dtype", "variable_dtype"):
                v = getattr(l, attr, None)
                if v is None:
                    continue
                try:
                    return v.name
                except Exception:
                    return str(v)
                return "?"

        def fmt_weights(weights):
            """Format a list of variables: show name + shape + dtype."""
            if not weights:
                return "[]"
            parts = []
             
            for w in weights:
                # w.dtype can be tf.DType; use .name if available
                wdt = getattr(w, "dtype", None)
                try:
                    wdt = wdt.name
                except Exception:
                    wdt = str(wdt) if wdt is not None else "?"
                parts.append(f"{w.name}: {tuple(w.shape)}:{wdt}")
                 
            return "[" + ", ".join(parts) + "]"
    
        
        #######################
        
        # Prefer .input / .output (KerasTensors), fall back to *_shape
        in_t = safe_attr(layer, "input")
        if in_t is None:
            in_t = safe_attr(layer, "inputs")

        out_t = safe_attr(layer, "output")
        if out_t is None:
            out_t = safe_attr(layer, "outputs")

        in_shape  = shape_from_tensor(in_t)  if (in_t  is not None) else shape_from_tensor(safe_attr(layer, "input_shape"))
        out_shape = shape_from_tensor(out_t) if (out_t is not None) else shape_from_tensor(safe_attr(layer, "output_shape"))

        # dtype reporting (prefer IO tensors; fall back to layer-level dtype)
        in_dtype = dtype_from_tensor(in_t) if (in_t is not None) else layer_dtype_fallback(layer)
        out_dtype = dtype_from_tensor(out_t) if (out_t is not None) else layer_dtype_fallback(layer)


        w_str = fmt_weights(layer.weights)
        # --- print this layer --------------------------------------------------------
        print(f"{pad}{layer.name} ({cls_name})  in={in_shape}:{in_dtype}   out={out_shape}:{out_dtype}   weights={w_str}")
        
        # --- recurse into sublayers --------------------------------------------------
        sublayers = getattr(layer, "layers", None)
        if sublayers:
            for sub in sublayers:
                # Avoid infinite recursion on Models that include themselves
                if sub is layer:
                    continue
                NetworkBuilder.recursive_summary(sub, indent + 2)

    @staticmethod
    def compatibility_checks(args):
        # Tokenizer
        if args.tokenizer:
            if args.tokenizer_max_tokens is None:
                handle_error("Tokenizer requires --tokenizer_max_tokens", args.verbose)
            if args.tokenizer_output_sequence_length is None:
                handle_error("Tokenizer requires --tokenizer_output_sequence_length", args.verbose)

        if args.embedding or args.tokenizer:
            if args.embedding_dimensions is None:
                handle_error("Tokenizer/Embeddding require --embedding_dimensions", args.verbose)
