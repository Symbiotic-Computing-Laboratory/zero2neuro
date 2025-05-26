import os
import sys

# TODO: need to fix this path
sys.path.append('../../../keras3_tools/src/')

from fully_connected_tools import *

class NetworkBuilder:
    
    @staticmethod
    def args2model(args):
        if args.network_type == 'fully_connected':
            model = create_fully_connected_network(input_shape=args.input_shape0,
                                               n_hidden=args.number_hidden_units,
                                               output_shape=args.output_shape0,
                                               dropout_input=args.dropout_input,
                                               name_base='',
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
            assert False, 'Unsupported network type (%s)'%args.network_type

        return model


