'''

Top-level Deep Neural Network Engine


'''

import os
import sys
import socket


from parser import *
from dataset import *
from network_builder import *

import wandb
from keras.utils import plot_model

def compatibility_checks():
    pass
    
def args2wandb_name(args)->str:
    # TODO: make generic like fbase
    outstr = args.experiment_name
    if args.rotation is not None:
        outstr = outstr + '_R%d'%args.rotation

    return outstr

def args2fbase(args):
    '''
    '''
    if args.output_file_base is None:
        # Generate default output file name
        outstr = '%s/%s'%(args.results_path, args.experiment_name)

        if args.conv_nfilters is not None:
            outstr = outstr + '_filt_' + '_'.join(str(x) for x in args.conv_nfilters)

        if args.number_hidden_units is not None:
            outstr = outstr + '_fc_' + '_'.join(str(x) for x in args.number_hidden_units)

        if args.rotation is not None:
            outstr = outstr + '_R%d'%args.rotation

    else:
        # Use output file name in the specified format
        outstr = '%s/%s'%(args.results_path, 
                          args.output_file_base.format(args=args))
        
    return outstr

    
def execute_exp(sds, model, args):
    # 
    if args.verbose >= 2:
        print(model.summary())

    # Output file base and pkl file
    fbase = args2fbase(args)
    print(fbase)
    fname_out = "%s_results.pkl"%fbase

    # Plot the model
    if args.render_model:
        render_fname = '%s_model_plot.png'%fbase
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        # No!
        print("NO GO")
        #print(fbase)
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #####
    if args.wandb:
        # Start wandb
        run = wandb.init(project=args.wandb_project,
                         name=args2wandb_name(args),
                         notes=fbase,
                         config=vars(args))

        # Log hostname
        wandb.log({'hostname': socket.gethostname()})

        # Log model design image
        if args.render_model:
            wandb.log({'model architecture': wandb.Image(render_fname)})

            
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
        
    # LOG RESULTS

    # List of evaluation measures
    eval_list = [args.loss]
    if args.metrics is not None:
        eval_list = eval_list + args.metrics

    results = {}
    ######
    # Training
    ev = model.evaluate(sds.ins_training,
                        sds.outs_training,
                        steps=args.steps_per_epoch)
    d = dict(zip(['training_'+s for s in eval_list], ev))
    
    results.update(d)
    
    if args.wandb:
        wandb.log(d)

    ######
    # Training set
    if args.log_training_set:
        # TODO: only works for not TF Datasets
        results['ins_training'] = sds.ins_training
        results['outs_training'] = sds.outs_training
        results['predict_training'] = model.predict(sds.ins_training)
                                                
    
    ######
    # Validation set
    if (sds.ins_validation is not None):
        ev = model.evaluate(sds.ins_validation,
                            sds.outs_validation)
        d = dict(zip(['validation_'+s for s in eval_list], ev))
    
        results.update(d)
    
        if args.wandb:
            wandb.log(d)

        if args.log_validation_set:
            # TODO: only works for not TF Datasets
            results['ins_validation'] = sds.ins_validation
            results['outs_validation'] = sds.outs_validation
            results['predict_validation'] = model.predict(sds.ins_validation)

    ######
    # Testing set
    if (sds.ins_testing is not None):
        ev = model.evaluate(sds.ins_testing,
                            sds.outs_testing)
        d = dict(zip(['testing_'+s for s in eval_list], ev))
    
        results.update(d)
    
        if args.wandb:
            wandb.log(d)

        if args.log_testing_set:
            # TODO: only works for not TF Datasets
            results['ins_testing'] = sds.ins_testing
            results['outs_testing'] = sds.outs_testing
            results['predict_testing'] = model.predict(sds.ins_testing)

    ######
    # Close WANDB
    if args.wandb:
        wandb.finish()

    # Save description of dataset
    results['dataset'] = sds.describe()

    # Save results
    results['fname_base'] = fbase
    results['args'] = args
    
    # Save history
    results['history'] = history.history

    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)

    # Save model
    if args.save_model:
        model.save("%s_model.keras"%(fbase))


def prepare_and_execute_experiment(args):
    # Compatibility checks
    compatibility_checks()
    
    ######
    # GPU configuration
    # Turn off GPU?
    # TODO: more general handling of CUDA_VISIBLE_DEVICES
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
    if not args.network_test:
        sds = SuperDataSet(args)

    ######
    # Create the model
    model = NetworkBuilder.args2model(args)

    if args.network_test:
        print(model.summary())
        # Don't go any further
        return

    
    ######
    # Execute the experiment
    execute_exp(sds, model, args)
    
if __name__ == "__main__":
    # Command line arguments
    parser = create_parser()
    args = parser.parse_args()

    print(args)

    prepare_and_execute_experiment(args)
    
