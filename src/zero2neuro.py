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

def compatibility_checks(args):
    if args.data_output_sparse_categorical:
        handle_error('data_output_sparse_categorical is no longer supported.  Use data_columns_categorical_to_int instead', args.verbose)

    if args.data_split is not None:
        handle_error('data_split is no longer supported.  Use data_fold_split and data_set_type', args.verbose)

    if args.n_folds is not None:
        handle_error('n_folds is no longer supported.  Use data_n_folds', args.verbose)
        
    if args.n_training_folds is not None:
        handle_error('n_training_folds is no longer supported.  Use data_n_training_folds', args.verbose)
        
    if args.training_mode is not None:
        handle_error('training_mode is no longer supported', args.verbose)

    if not args.data_representation in ['numpy', 'tf-dataset']:
        handle_error("data_representation must be either numpy or tf-dataset", args.verbose)

    if args.rotation is not None:
        handle_error("rotation is expired.  Use data_rotation instead", args.verbose)

    # reporting relies on logging
    if args.report_training and not args.log_training:
        handle_error("If report_training=True, then log_training must also be True", args.verbose)

    if args.report_validation and not args.log_validation:
        handle_error("If report_validation=True, then log_validation must also be True", args.verbose)

    if args.report_testing and not args.log_testing:
        handle_error("If report_testing=True, then log_testing must also be True", args.verbose)

    # reporting_ins relies on reporting
    if args.report_training_ins and not args.report_training:
        handle_error("If report_training_ins=True, then report_training must also be True", args.verbose)

    if args.report_validation_ins and not args.report_validation:
        handle_error("If report_validation_in=True, then report_validation must also be True", args.verbose)

    if args.report_testing_ins and not args.report_testing:
        handle_error("If report_testing_in=True, then report_testing must also be True", args.verbose)

    
def args2wandb_name(args)->str:
    #outstr = args.experiment_name
    #if args.rotation is not None:
    #outstr = outstr + '_R%d'%args.rotation

    try:
        output_name = args.wandb_name.format(args=args)
    except (ValueError, AttributeError, KeyError) as e:
        print(e)
        handle_error('Error: args.wandb_name (%s)'%
                         args.wandb_name, args.debug)
            
    return output_name


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
        # TODO: need to check that the format string is valid
        try:
            output_file_base = args.output_file_base.format(args=args)
        except (ValueError, AttributeError, KeyError) as e:
            print(e)
            handle_error('Error: args.output_file_base cannot be parsed (%s)'%
                         args.output_file_base, args.debug)
            
        outstr = '%s/%s'%(args.results_path, output_file_base)
        
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

    if args.early_stopping:
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

    if args.epochs > 0:
        # Train model
        if args.data_format == 'tf-dataset':
            # TF-Datasets
            history = model.fit(sds.training,
                                steps_per_epoch=args.steps_per_epoch,
                                epochs=args.epochs,
                                validation_data=sds.validation,
                                verbose=args.verbose>=3,
                                callbacks=cbs)
        else:
            # Numpy arrays
            history = model.fit(sds.ins_training,
                                sds.outs_training,
                                sample_weight=sds.weights_training,
                                steps_per_epoch=args.steps_per_epoch,
                                epochs=args.epochs,
                                batch_size=args.batch,
                                validation_data=sds.validation,
                                verbose=args.verbose>=3,
                                callbacks=cbs)

    #######
    # LOG RESULTS

    # List of evaluation measures
    eval_list = [args.loss]
    if args.metrics is not None:
        eval_list = eval_list + args.metrics

    results = {}
    ######
    # Training
    print_debug('Training eval', 4, args.debug)
    if args.data_format == 'tf-dataset':
        ev = model.evaluate(sds.training,
                            steps=args.steps_per_epoch,
                            batch_size=args.batch,
                            )
    else:
        ev = model.evaluate(sds.ins_training,
                            sds.outs_training,
                            steps=args.steps_per_epoch,
                            batch_size=args.batch,
                            )
    d = dict(zip(['training_'+s for s in eval_list], ev))
    results.update(d)
    
    if args.wandb:
        wandb.log(d)

    ######
    # Training set
    if args.log_training_set and args.data_representation == 'numpy':
        print_debug('Training predict', 4, args.debug)
        # TODO: only works for not TF Datasets
        results['ins_training'] = sds.ins_training
        results['outs_training'] = sds.outs_training
        results['predict_training'] = model.predict(sds.ins_training)
    
    ######
    # Validation set
    if (sds.ins_validation is not None) or (sds.validation is not None):
        if args.data_format == 'tf-dataset':
            ev = model.evaluate(sds.validation)
        else:
            ev = model.evaluate(sds.ins_validation,
                                sds.outs_validation)
        d = dict(zip(['validation_'+s for s in eval_list], ev))
    
        results.update(d)
    
        if args.wandb:
            wandb.log(d)

        if args.log_validation_set and args.data_representation == 'numpy':
            # TODO: only works for not TF Datasets
            results['ins_validation'] = sds.ins_validation
            results['outs_validation'] = sds.outs_validation
            results['predict_validation'] = model.predict(sds.ins_validation)

    ######
    # Testing set
    if (sds.ins_testing is not None) or (sds.testing is not None):
        if args.data_format == 'tf-dataset':
            ev = model.evaluate(sds.testing)
        else:
            ev = model.evaluate(sds.ins_testing,
                                sds.outs_testing)
        d = dict(zip(['testing_'+s for s in eval_list], ev))
    
        results.update(d)
    
        if args.wandb:
            wandb.log(d)

        if args.log_test_set and args.data_representation == 'numpy':
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

    if args.report:
        # Creates a writer for excel files.
        writer = pd.ExcelWriter("%s_results.xlsx"%(fbase), engine='xlsxwriter')
        
        if args.log_training_set and args.report_training :
            # Find out how many columns for to designate for each key and then append them to a list
            predict_columns = []
            for i in range(results['predict_training'].shape[1]):
                predict_columns.append('Prediction_%i' % i)
        
            # Make a dataframe for each metric and designate the correct number of columns. (i.e [5, 6, 9] would need 3 columns)
            df_tr_ins = pd.DataFrame(results['ins_training'], columns=args.data_inputs)
            df_tr_outs = pd.DataFrame(results['outs_training'], columns=args.data_outputs)
            df_tr_predict = pd.DataFrame(results['predict_training'], columns=predict_columns)
    
            # Combine the dataframes into one.
            df_combined_training = pd.concat([df_tr_ins, df_tr_outs, df_tr_predict], axis=1)
    
            # Write the dataframe to the appropiate sheet in the excel file (This one is for training)
            df_combined_training.to_excel(writer, sheet_name='Training Data', index=False)
    
        if args.log_validation_set and args.report_validation:
            predict_columns = []
            for i in range(results['predict_validation'].shape[1]):
                predict_columns.append('Prediction_%i' % i)
            df_val_ins = pd.DataFrame(results['ins_validation'], columns=args.data_inputs)
            df_val_outs = pd.DataFrame(results['outs_validation'], columns=args.data_outputs)
            df_val_predict = pd.DataFrame(results['predict_validation'], columns=predict_columns)
            df_combined_validation = pd.concat([df_val_ins, df_val_outs, df_val_predict], axis=1)
            df_combined_validation.to_excel(writer, sheet_name='Validation Data', index=False)
    
        if args.log_test_set and args.report_testing:
            predict_columns = []
            for i in range(results['predict_testing'].shape[1]):
                predict_columns.append('Prediction_%i' % i)
            df_test_ins = pd.DataFrame(results['ins_testing'], columns=args.data_inputs)
            df_test_outs = pd.DataFrame(results['outs_testing'], columns=args.data_outputs)
            df_test_predict = pd.DataFrame(results['predict_testing'], columns=predict_columns)
            df_combined_validation = pd.concat([df_test_ins, df_test_outs, df_test_predict], axis=1)
            df_combined_validation.to_excel(writer, sheet_name='Testing Data', index=False)
    
        # Create the excel file and save it.
        writer.close()

    # Save model
    if args.save_model:
        model.save("%s_model.keras"%(fbase))


def prepare_and_execute_experiment(args):
    # Compatibility checks
    compatibility_checks(args)
    
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
    #if not args.network_test:
    sds = SuperDataSet(args)

    ######
    # Create the model
    model = NetworkBuilder.args2model(args)

    #if args.network_test:
    #print(model.summary())
    #    # Don't go any further
    #    return

    
    ######
    # Execute the experiment
    execute_exp(sds, model, args)
    
if __name__ == "__main__":
    # Command line arguments
    parser = create_parser()
    args = parser.parse_args()

    print(args)

    prepare_and_execute_experiment(args)
    
