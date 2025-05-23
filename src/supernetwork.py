import os
import sys

# TODO: need a better way to handle this (this path is relative to the executing directory, not the src directory)
sys.path.append('../../../keras3_tools/src/')


from parser import *
from dataset import *
from fully_connected_tools import *
import wandb

# TODO: push into its own class
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



def execute_exp(sds, model, args):
    # 
    if args.verbose >= 2:
        print(model.summary())

    # Output file base and pkl file
    #fbase = generate_fname(args, args_str)
    #print(fbase)
    #fname_out = "%s_results.pkl"%fbase

    # Plot the model
    #if args.render:
    #render_fname = '%s_model_plot.png'%fbase
     #   plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        #print(fbase)
        return

    # Check if output file already exists
    #if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
    #    print("File %s already exists"%fname_out)
    #    return

    #####
    #if args.wandb:
    #    # Start wandb
    #    run = wandb.init(project=args.project, name='%s_R%d'%(args.label,args.rotation), notes=fbase, config=vars(args))

        # Log hostname
    #    wandb.log({'hostname': socket.gethostname()})

        # Log model design image
    #    if args.render:
    #        wandb.log({'model architecture': wandb.Image(render_fname)})

            
    #####
    # Callbacks
    cbs = []
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.early_stopping_patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.early_stopping_min_delta,
                                                      monitor=args.early_stopping_monitor)
    cbs.append(early_stopping_cb)

    if args.wandb:
        # Weights and Biases
        wandb_metrics_cb = wandb.keras.WandbMetricsLogger()
        cbs.append(wandb_metrics_cb)

    if args.verbose >= 1:
        print('Fitting model')

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used
    
    history = model.fit(sds.ins_training,
                        sds.outs_training,
                        steps_per_epoch=args.steps_per_epoch,
                        epochs=args.epochs, 
                        validation_data=sds.validation,
                        verbose=args.verbose>=3,
                        callbacks=cbs)
        
    # TOOD: log results
    # TODO: save model
    
    
if __name__ == "__main__":
    # Command line arguments
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    ######
    # GPU configuration
    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    # GPU check
    visible_devices = tf.config.get_visible_devices('GPU') 
    n_visible_devices = len(visible_devices)
    print('GPUS:', visible_devices)
    if n_visible_devices > 0:
        for device in visible_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print('We have %d GPUs\n'%n_visible_devices)
    else:
        print('NO GPU')


    ######
    # Fetch the dataset
    sds = SuperDataSet(args)
    print(sds.outs_training, sds.ins_training)

    ######
    # Create the model
    model=args2model(args)

    
    ######
    # Execute the experiment
    execute_exp(sds, model, args)
    
