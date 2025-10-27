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


VERSION = "0.1"
GITHUB = "https://github.com/Symbiotic-Computing-Laboratory/zero2neuro"

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

    # reporting_ins relies on reporting
    if args.report_training_ins and not args.report_training:
        handle_error("If report_training_ins=True, then report_training must also be True", args.verbose)

    if args.report_validation_ins and not args.report_validation:
        handle_error("If report_validation_in=True, then report_validation must also be True", args.verbose)

    if args.report_testing_ins and not args.report_testing:
        handle_error("If report_testing_in=True, then report_testing must also be True", args.verbose)

    # Check tabular arguments
    if args.tabular_column_range is not None and args.tabular_column_list is not None:
        handle_error("Can only provide at most one of tabular_column_range or tabular_column_list", args.verbose)

    # Check that results path exists
    if not os.path.exists(args.results_path):
        handle_error("results_path must exist", args.verbose)
        
    if not os.path.isdir(args.results_path):
        handle_error("results_path must be a directory", args.verbose)

    # Early stopping checks
    if (not args.early_stopping) and (args.early_stopping_monitor or args.early_stopping_patience):
        handle_error("You must use the early_stopping argument if you specify either early_stopping_monitor or early_stopping_patience", args.verbose)
    

    
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
        except (ValueError, AttributeError, TypeError, KeyError) as e:
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
        metrics_list = {}
    

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
    
    if args.report:
        metrics_list.update(d)
    
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
        print('test')
        if args.data_format == 'tf-dataset':
            ev = model.evaluate(sds.validation)
        else:
            ev = model.evaluate(sds.ins_validation,
                                sds.outs_validation)
        d = dict(zip(['validation_'+s for s in eval_list], ev))
    
        results.update(d)

        if args.report:
            metrics_list.update(d)
        
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

        if args.report:
            metrics_list.update(d)
    
        if args.wandb:
            wandb.log(d)

        if args.log_testing_set and args.data_representation == 'numpy':
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

    #TODO: Add support for tf-datasets 
    if args.report:
        # Creates a writer for excel files.
        writer = pd.ExcelWriter("%s_report.xlsx"%(fbase), engine='xlsxwriter')

        # Create a sheet for key arguments and all arguments. 
        df_key_args_list, df_args_list = xlsx_args_list(args)

        # Sheet 1: Key Arguments
        df_key_args_list.to_excel(writer, sheet_name='Key Arguments List', index=False)
        

        # Create sheet for the loss and metrics.
        df_performance_report = xlsx_performance_report(metrics_list)

        # Sheet 2: Performance Report
        df_performance_report.to_excel(writer, sheet_name='Performance Report', index=False)

        # Sheet 3: All Arguments
        df_args_list.to_excel(writer, sheet_name='Arguments List', index=False)
        
        # Create a sheet for each training/validation/testing set and their true values vs predictions.
        if args.report_training:
            
            df_training_report = xlsx_training_report(sds,model, args)
            
            df_training_report.to_excel(writer, sheet_name='Training Data', index=False)
    
        if args.report_validation and not args.data_format == 'tf-dataset':

            df_validation_report = xlsx_validation_report(sds,model, args)
            
            df_validation_report.to_excel(writer, sheet_name='Validation Data', index=False)
    
        if args.report_testing and not args.data_format == 'tf-dataset':
            
            df_testing_report = xlsx_testing_report(sds,model, args)
            
            df_testing_report.to_excel(writer, sheet_name='Testing Data', index=False)
    
        # Create the excel file and save it.
        writer.close()

    # Save model
    if args.save_model:
        model.save("%s_model.keras"%(fbase))

def xlsx_args_list(args):
    '''
    Function that creates two pandas dataframes for the arguments list

    Args:
        args(argparse.Namespace): The list of arguments the user has defined.
    Returns:
        Two dataframes, one with only the key arguments and the other with all the arguments
        from the parser.
    '''
    # Turn the args list into a dictionary
    args_dict = args.__dict__
    
    # We're going to change all the values be lists to ensure consistency 
    normalized_args_dict = {}
    for key, value in args_dict.items():
        if not isinstance(value, list):
            normalized_args_dict[key] = [value]
        else:
            normalized_args_dict[key] = value
    
    # Grab the values for dictionary and assign to a variable for simplicity
    values = normalized_args_dict.values()
    
    # DataFrames don't like it when values aren't the same length, so we'll pad the values so all keys have the same amount.
    padded_values = list(zip_longest(*values, fillvalue = None))
    
    # Make the dataframe where the column header is the key and the rows contain the values for that key.
    df_args = pd.DataFrame(padded_values, columns=normalized_args_dict.keys())

    # Make a key arguments dataframe off full arguments dataframe
    df_key_args = df_args[['experiment_name', 'output_file_base', 'dataset_directory',
                           'data_n_folds','data_rotation', 'data_files', 
                           'data_groups', 'wandb', 'network_type']]
    
    # If a column is missing values, just remove the column.
    df_args = df_args.dropna(axis=1, how='all')
    
    # Replace the null values with an empty space for readability
    df_args = df_args.fillna('')
    
    
    return(df_key_args, df_args)

def xlsx_performance_report(merged_metrics):
    '''
    Function creates a pandas dataframe for performance metrics.

    Args: 
        merged_metrics(dict): A dictionary with train/val/test metrics merged if applicable.

    Returns:
        Pandas DataFrame with one row of data for each column of training/validation/testing metric. 
    '''

    keys = merged_metrics.keys()
    values = merged_metrics.values()

    df_performance_rep = pd.DataFrame([values], columns=keys)

    return(df_performance_rep)

def xlsx_training_report(sds, model, args):
    '''
    Function creates a pandas dataframe for the training dataset.

    Args:
        sds(dataset.SuperDataSet'): The object the contains the processed data 
                                    used for training/evaluating the model.
        model(keras.src.models.functional.Functional): The trained Keras model.
        args(argparse.Namespace): The list of arguments the user has defined. 

    Returns:
        A pandas dataframe that contains the predictions and true values 
        of the training dataset and the training data if applicable.
    '''
    
    predict_columns = []
    
    
    if args.data_representation == 'numpy':
        outs = sds.outs_training
        predictions = model.predict(sds.ins_training)
    else:
        outs = sds.training.map(lambda x, y: y)
        outs = [label.numpy() for label in outs]
        outs = np.concatenate(outs, axis=0)
        training_prediction_set = sds.training.map(lambda x, y: x)
        predictions = model.predict(training_prediction_set)

    for i in range(predictions.shape[1]):
        predict_columns.append('Prediction_%i' % i)
         
    df_outs = pd.DataFrame(outs, columns=args.data_outputs)
    df_predict = pd.DataFrame(predictions, columns=predict_columns)

    # Combine the dataframes into one.
    df_combined = pd.concat([df_outs, df_predict], axis=1)

    if args.report_training_ins:
        ins = sds.ins_training
        df_ins = pd.DataFrame(ins, columns=args.data_inputs)
        df_combined = pd.concat([df_combined, df_ins], axis=1)
        

    return(df_combined)
    
def xlsx_validation_report(sds, model, args):
    '''
    Function creates a pandas dataframe for the validation dataset.

    Args:
        sds(dataset.SuperDataSet'): The object the contains the processed data 
                                    used for training/evaluating the model.
        model(keras.src.models.functional.Functional): The trained Keras model.
        args(argparse.Namespace): The list of arguments the user has defined. 

    Returns:
        A pandas dataframe that contains the predictions and true values 
        of the validation dataset and the validation data if applicable.
    '''
    
    predict_columns = []
    
    outs = sds.outs_validation
    predictions = model.predict(sds.ins_validation)

    for i in range(predictions.shape[1]):
        predict_columns.append('Prediction_%i' % i)
         
    df_outs = pd.DataFrame(outs, columns=args.data_outputs)
    df_predict = pd.DataFrame(predictions, columns=predict_columns)

    # Combine the dataframes into one.
    df_combined = pd.concat([df_outs, df_predict], axis=1)

    if args.report_validation_ins:
        ins = sds.ins_validation
        df_ins = pd.DataFrame(ins, columns=args.data_inputs)
        df_combined = pd.concat([df_combined, df_ins], axis=1)
    
    return(df_combined)
    
def xlsx_testing_report(sds, model, args):
    '''
    Function creates a pandas dataframe for the testing dataset.

    Args:
        sds(dataset.SuperDataSet'): The object the contains the processed data 
                                    used for training/evaluating the model.
        model(keras.src.models.functional.Functional): The trained Keras model.
        args(argparse.Namespace): The list of arguments the user has defined. 

    Returns:
        A pandas dataframe that contains the predictions and true values 
        of the testing dataset and the testing data if applicable.
    '''
    
    predict_columns = []
    
    outs = sds.outs_testing
    predictions = model.predict(sds.ins_testing)

    for i in range(predictions.shape[1]):
        predict_columns.append('Prediction_%i' % i)
         
    df_outs = pd.DataFrame(outs, columns=args.data_outputs)
    df_predict = pd.DataFrame(predictions, columns=predict_columns)

    # Combine the dataframes into one.
    df_combined = pd.concat([df_outs, df_predict], axis=1)

    if args.report_testing_ins:
        ins = sds.ins_testing
        df_ins = pd.DataFrame(ins, columns=args.data_inputs)
        df_combined = pd.concat([df_combined, df_ins], axis=1)

    return(df_combined)


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

    n = len(GITHUB)
    ver = "This is Zero2Neuro Version " + VERSION
    print("\n"
          + "*" * n
          + "\n"
          + " " * ((n-len(ver))//2)
          + ver
          + "\n\n"
          + GITHUB
          + "\n"
          + "*" * n
          + "\n")

    print(args)

    prepare_and_execute_experiment(args)
    
