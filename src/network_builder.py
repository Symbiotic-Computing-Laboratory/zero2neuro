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
                model = FullyConnectedNetwork.create_fully_connected_network(input_shape=args.input_shape0,
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
                                                                             metrics=args.metrics)
            elif args.network_type == 'cnn':
                model = ConvolutionalNeuralNetwork.create_cnn_network(input_shape=args.input_shape0,
                                                                      conv_kernel_size=args.conv_kernel_size,
                                                                      conv_padding=args.conv_padding,
                                                                      conv_number_filters=args.conv_number_filters,
                                                                      conv_activation=args.conv_activation,
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
                                                                      metrics=args.metrics)
                                               
            else:
                handle_error('Unsupported network type (%s)'%args.network_type, args.verbosity)

        return model


