"""
Registry mapping pipeline-element names -> sklearn constructor + lazy args.

Each registry entry is a dict with:
    constructor : the sklearn class to instantiate
    args        : a callable  ns -> {kwarg: value}, evaluated ONLY when called
    checks      : (optional) a set of namespace attributes that must exist
                  *before* any argument is evaluated

Because `args` is a callable, the `ns.<attr>` lookups inside it are deferred
until build time -- they are not evaluated when the registry is defined.
"""

from argparse import ArgumentParser, Namespace

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from dataset import *
from zero2neuro_debug import *

class SklearnModeler:
    PIPELINE_REGISTRY = {
        # ---- normalized feature transformations (their own pipeline steps) ----
        "standard_scaler": {
            "constructor": StandardScaler,
            "args": lambda ns: {
                #"with_mean": ns.scaler_with_mean,
                #"with_std": ns.scaler_with_std,
            },
            "checks": {}, #"scaler_with_mean", "scaler_with_std"},
        },
        "minmax_scaler": {
            "constructor": MinMaxScaler,
            "args": lambda ns: {
                #"feature_range": (ns.minmax_low, ns.minmax_high),
            },
            "checks": {}, #"minmax_low", "minmax_high"},
        },

        "polynomial": {
            "constructor": PolynomialFeatures,
            "args": lambda ns: {
                "degree": ns.skl_poly_degree,
                "interaction_only": ns.skl_poly_interaction_only,
                "include_bias": ns.skl_include_bias,
            },
            "checks": {"skl_poly_degree", "skl_poly_interaction_only", "skl_include_bias"},
        },

        # ---- estimators ----
        "linear_regression": {
            "constructor": LinearRegression,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
                #"positive": ns.linreg_positive,
            },
            "checks": {"skl_include_bias"}, #, "linreg_positive"},
        },
        "ridge_regression": {
            "constructor": Ridge,  # sklearn's class is `Ridge`, not `RidgeRegression`
            "args": lambda ns: {
                "alpha": ns.L2_regularization,
                "fit_intercept": ns.skl_include_bias,
                "solver": ns.skl_solver,
                "max_iter": ns.skl_max_iter,
            },
            "checks": {"L2_regularization"},
        },
    }

    def __init__(self, args:ArgumentParser, fbase:str):
        '''
        '''
        self.args = args
        self.fbase = fbase
        self.pipeline = self.build_pipeline(args)

    def execute_exp(self, sds:SuperDataSet):
        
        #if self.args.verbose >= 2:
        #    print(model.summary())

        # Results file
        fname_out = "%s_results.pkl"%self.fbase

        # Plot the model
        #if args.render_model:
        #    render_fname = '%s_model_plot.png'%self.fbase
        #    plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

        # Perform the experiment?
        if self.args.nogo:
            # No!
            print("NO GO")
            return

        # Check if output file already exists
        if not self.args.force and os.path.exists(fname_out):
            # Results file does exist: exit
            print("File %s already exists"%fname_out)
            return
        
        if self.args.verbose >= 1:
            print('Fitting model')

        # Train the model
        self.pipeline.fit(sds.ins_training, sds.outs_training)

        # Log the results
        results = {}

        # Training
        ev = self.pipeline.score(X=sds.ins_training, y=sds.outs_training)
        results['score_training'] = ev

        # Log the details?
        if self.args.log_training_set:
            print_debug('Training predict', 4, self.args.debug)
            results['ins_training'] = sds.ins_training
            results['outs_training'] = sds.outs_training
            results['predict_training'] = self.pipeline.predict(sds.ins_training)

        # Validation Data Set
        if sds.ins_validation is not None:
            ev = self.pipeline.score(X=sds.ins_validation, y=sds.outs_validation)
            results['score_validation'] = ev

            # Log the details?
            if self.args.log_validation_set:
                print_debug('Validation predict', 4, self.args.debug)
                results['ins_validation'] = sds.ins_validation
                results['outs_validation'] = sds.outs_validation
                results['predict_validation'] = self.pipeline.predict(sds.ins_validation)
                

        # Testing Data Set
        if sds.ins_testing is not None:
            ev = self.pipeline.score(X=sds.ins_testing, y=sds.outs_testing)
            results['score_testing'] = ev

            # Log the details?
            if self.args.log_testing_set:
                print_debug('Testing predict', 4, self.args.debug)
                results['ins_testing'] = sds.ins_testing
                results['outs_testing'] = sds.outs_testing
                results['predict_testing'] = self.pipeline.predict(sds.ins_testing)
                
        # Save description of dataset
        results['dataset'] = sds.describe()

        # Save results
        results['fname_base'] = self.fbase
        results['args'] = self.args

        # Write out results
        with open("%s_results.pkl"%(self.fbase), "wb") as fp:
            pickle.dump(results, fp)

        # TODO: Luke: add report generation (as a separate method call)

        # Save model
        if self.args.save_model:
            with open("%s_model.pkl"%(self.fbase), "wb") as fp:
                model = {'model': self.pipeline}
                pickle.dump(model, fp)


    def build_step(self, name: str, ns: Namespace):
        '''
        Instantiate a single pipeline element from the namespace.

        Existence checks run first; constructor args are evaluated only afterward,
        at the moment the constructor is called.
        '''

        # name -> details of the sklearn component
        entry = SklearnModeler.PIPELINE_REGISTRY[name]

        # Check for missing args
        missing = {a for a in entry.get("checks", set()) if (not hasattr(ns, a)) or (getattr(ns, a) is None)}
        if missing:
            handle_error(f"pipeline element {name!r}: missing argument(s) {sorted(missing)}", self.args.verbose)

        # Get the arguments for the constructor
        kwargs = entry["args"](ns)          # <-- lazy: ns.<attr> read only here

        # Create and return the pipeline element
        return entry["constructor"](**kwargs)

    def build_pipeline(self, ns: Namespace) -> Pipeline:
        '''
        Build a Pipeline from ns.pipeline (an ordered list of element names).
        '''

        return Pipeline([(name, self.build_step(name, ns)) for name in ns.skl_pipeline])
