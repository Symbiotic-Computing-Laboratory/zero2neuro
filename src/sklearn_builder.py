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

import numpy as np
from argparse import ArgumentParser, Namespace

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeClassifier,
    Lasso, ElasticNet, LassoLarsIC,
    Lars, LassoLars, OrthogonalMatchingPursuit,
    ARDRegression, BayesianRidge,
    MultiTaskElasticNet, MultiTaskLasso,
    HuberRegressor, QuantileRegressor,
    RANSACRegressor, TheilSenRegressor,
    GammaRegressor, PoissonRegressor, TweedieRegressor,
    LogisticRegression, Perceptron,
    SGDClassifier, SGDRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PolynomialFeatures,
    MaxAbsScaler, Normalizer, PowerTransformer,
    QuantileTransformer, RobustScaler, SplineTransformer,
)
from sklearn.cluster import (
    AffinityPropagation, AgglomerativeClustering, Birch,
    BisectingKMeans, DBSCAN, FeatureAgglomeration,
    HDBSCAN, KMeans, MeanShift, MiniBatchKMeans,
    SpectralBiclustering, SpectralClustering, SpectralCoclustering,
)
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreeRegressor,
)
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    IsolationForest, RandomForestClassifier, RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn.impute import (
    KNNImputer, MissingIndicator, SimpleImputer,
)
from sklearn.manifold import (
    Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.naive_bayes import (
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB,
)
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR
from sklearn.decomposition import (
    FactorAnalysis, FastICA, IncrementalPCA, KernelPCA,
    LatentDirichletAllocation, NMF, MiniBatchNMF,
    MiniBatchSparsePCA, PCA, SparseCoder, SparsePCA, TruncatedSVD,
)
from sklearn.pipeline import Pipeline

from dataset import *
from zero2neuro_debug import *


class TransformOutputRavel(BaseEstimator, TransformerMixin):
    """Ravel the input array to 1-D; used in the y-pipeline to flatten (N,1) targets."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.ravel(X)


class SklearnModeler:
    PIPELINE_REGISTRY = {
        # ---- normalized feature transformations (their own pipeline steps) ----
        "StandardScaler": {
            "constructor": StandardScaler,
            "args": lambda ns: {
                #"with_mean": ns.scaler_with_mean,
                #"with_std": ns.scaler_with_std,
            },
            "checks": {}, #"scaler_with_mean", "scaler_with_std"},
        },
        "MinMaxScaler": {
            "constructor": MinMaxScaler,
            "args": lambda ns: {
                #"feature_range": (ns.minmax_low, ns.minmax_high),
            },
            "checks": {}, #"minmax_low", "minmax_high"},
        },

        "PolynomialFeatures": {
            "constructor": PolynomialFeatures,
            "args": lambda ns: {
                "degree": ns.skl_poly_degree,
                "interaction_only": ns.skl_poly_interaction_only,
                "include_bias": ns.skl_include_bias,
            },
            "checks": {"skl_poly_degree", "skl_poly_interaction_only", "skl_include_bias"},
        },
        "MaxAbsScaler": {
            "constructor": MaxAbsScaler,
            "args": lambda ns: {},
            "checks": {},
        },
        "Normalizer": {
            "constructor": Normalizer,
            "args": lambda ns: {
                **({} if ns.skl_norm is None else {"norm": ns.skl_norm}),
            },
            "checks": {},
        },
        "PowerTransformer": {
            "constructor": PowerTransformer,
            "args": lambda ns: {
                **({} if ns.skl_method is None else {"method": ns.skl_method}),
                **({} if ns.skl_standardize is None else {"standardize": ns.skl_standardize}),
            },
            "checks": {},
        },
        "QuantileTransformer": {
            "constructor": QuantileTransformer,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_quantiles is None else {"n_quantiles": ns.skl_n_quantiles}),
                **({} if ns.skl_output_distribution is None else {"output_distribution": ns.skl_output_distribution}),
            },
            "checks": {},
        },
        "RobustScaler": {
            "constructor": RobustScaler,
            "args": lambda ns: {
                **({} if ns.skl_with_centering is None else {"with_centering": ns.skl_with_centering}),
                **({} if ns.skl_with_scaling is None else {"with_scaling": ns.skl_with_scaling}),
            },
            "checks": {},
        },
        "SplineTransformer": {
            "constructor": SplineTransformer,
            "args": lambda ns: {
                "include_bias": ns.skl_include_bias,
                **({} if ns.skl_n_knots is None else {"n_knots": ns.skl_n_knots}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
            },
            "checks": {},
        },

        "TransformOutputRavel": {
            "constructor": TransformOutputRavel,
            "args": lambda ns: {},
            "checks": {},
        },

        # ---- estimators: least-squares / regularized regression ----
        "LinearRegression": {
            "constructor": LinearRegression,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
            },
            "checks": {"skl_include_bias"},
        },
        "Ridge": {
            "constructor": Ridge,
            "args": lambda ns: {
                "alpha": ns.L2_regularization,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
            },
            "checks": {"L2_regularization"},
        },
        "RidgeClassifier": {
            "constructor": RidgeClassifier,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "Lasso": {
            "constructor": Lasso,
            "args": lambda ns: {
                "alpha": ns.L1_regularization,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {"L1_regularization"},
        },
        "ElasticNet": {
            "constructor": ElasticNet,
            "args": lambda ns: {
                "alpha": ns.L1_regularization,
                "l1_ratio": ns.skl_l1_ratio,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {"L1_regularization", "skl_l1_ratio"},
        },
        "MultiTaskLasso": {
            "constructor": MultiTaskLasso,
            "args": lambda ns: {
                "alpha": ns.L1_regularization,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {"L1_regularization"},
        },
        "MultiTaskElasticNet": {
            "constructor": MultiTaskElasticNet,
            "args": lambda ns: {
                "alpha": ns.L1_regularization,
                "l1_ratio": ns.skl_l1_ratio,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {"L1_regularization", "skl_l1_ratio"},
        },

        # ---- estimators: least-angle regression ----
        "Lars": {
            "constructor": Lars,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
            },
            "checks": {},
        },
        "LassoLars": {
            "constructor": LassoLars,
            "args": lambda ns: {
                "alpha": ns.L1_regularization,
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
            },
            "checks": {"L1_regularization"},
        },
        "LassoLarsIC": {
            "constructor": LassoLarsIC,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
            },
            "checks": {},
        },
        "OrthogonalMatchingPursuit": {
            "constructor": OrthogonalMatchingPursuit,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_n_nonzero_coefs is None else {"n_nonzero_coefs": ns.skl_n_nonzero_coefs}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- estimators: Bayesian ----
        "BayesianRidge": {
            "constructor": BayesianRidge,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "ARDRegression": {
            "constructor": ARDRegression,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- estimators: robust / robust regression ----
        "HuberRegressor": {
            "constructor": HuberRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_epsilon is None else {"epsilon": ns.skl_epsilon}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "RANSACRegressor": {
            "constructor": RANSACRegressor,
            "args": lambda ns: {},
            "checks": {},
        },
        "TheilSenRegressor": {
            "constructor": TheilSenRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- estimators: quantile / GLM ----
        "QuantileRegressor": {
            "constructor": QuantileRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_quantile is None else {"quantile": ns.skl_quantile}),
                **({} if ns.L1_regularization is None else {"alpha": ns.L1_regularization}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
            },
            "checks": {},
        },
        "PoissonRegressor": {
            "constructor": PoissonRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "GammaRegressor": {
            "constructor": GammaRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "TweedieRegressor": {
            "constructor": TweedieRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_tweedie_power is None else {"power": ns.skl_tweedie_power}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- estimators: linear classifiers / SGD ----
        "LogisticRegression": {
            "constructor": LogisticRegression,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_penalty is None else {"penalty": ns.skl_penalty}),
                **({} if ns.skl_solver is None else {"solver": ns.skl_solver}),
                **({} if ns.skl_l1_ratio is None else {"l1_ratio": ns.skl_l1_ratio}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "Perceptron": {
            "constructor": Perceptron,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_eta0 is None else {"eta0": ns.skl_eta0}),
                **({} if ns.skl_penalty is None else {"penalty": ns.skl_penalty}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "SGDClassifier": {
            "constructor": SGDClassifier,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_loss is None else {"loss": ns.skl_loss}),
                **({} if ns.skl_penalty is None else {"penalty": ns.skl_penalty}),
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_l1_ratio is None else {"l1_ratio": ns.skl_l1_ratio}),
                **({} if ns.skl_eta0 is None else {"eta0": ns.skl_eta0}),
                **({} if ns.skl_learning_rate is None else {"learning_rate": ns.skl_learning_rate}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "SGDRegressor": {
            "constructor": SGDRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_loss is None else {"loss": ns.skl_loss}),
                **({} if ns.skl_penalty is None else {"penalty": ns.skl_penalty}),
                **({} if ns.L2_regularization is None else {"alpha": ns.L2_regularization}),
                **({} if ns.skl_l1_ratio is None else {"l1_ratio": ns.skl_l1_ratio}),
                **({} if ns.skl_eta0 is None else {"eta0": ns.skl_eta0}),
                **({} if ns.skl_learning_rate is None else {"learning_rate": ns.skl_learning_rate}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "PassiveAggressiveRegressor": {
            "constructor": PassiveAggressiveRegressor,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- clustering ----
        "AffinityPropagation": {
            "constructor": AffinityPropagation,
            "args": lambda ns: {
                **({} if ns.skl_damping is None else {"damping": ns.skl_damping}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
            },
            "checks": {},
        },
        "AgglomerativeClustering": {
            "constructor": AgglomerativeClustering,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
                **({} if ns.skl_linkage is None else {"linkage": ns.skl_linkage}),
            },
            "checks": {},
        },
        "Birch": {
            "constructor": Birch,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_threshold is None else {"threshold": ns.skl_threshold}),
                **({} if ns.skl_branching_factor is None else {"branching_factor": ns.skl_branching_factor}),
            },
            "checks": {},
        },
        "BisectingKMeans": {
            "constructor": BisectingKMeans,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "DBSCAN": {
            "constructor": DBSCAN,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_eps is None else {"eps": ns.skl_eps}),
                **({} if ns.skl_min_samples is None else {"min_samples": ns.skl_min_samples}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
            },
            "checks": {},
        },
        "FeatureAgglomeration": {
            "constructor": FeatureAgglomeration,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
                **({} if ns.skl_linkage is None else {"linkage": ns.skl_linkage}),
            },
            "checks": {},
        },
        "HDBSCAN": {
            "constructor": HDBSCAN,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_min_cluster_size is None else {"min_cluster_size": ns.skl_min_cluster_size}),
                **({} if ns.skl_min_samples is None else {"min_samples": ns.skl_min_samples}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
            },
            "checks": {},
        },
        "KMeans": {
            "constructor": KMeans,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "MeanShift": {
            "constructor": MeanShift,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_bandwidth is None else {"bandwidth": ns.skl_bandwidth}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
            },
            "checks": {},
        },
        "MiniBatchKMeans": {
            "constructor": MiniBatchKMeans,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_batch_size is None else {"batch_size": ns.skl_batch_size}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "SpectralClustering": {
            "constructor": SpectralClustering,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
            },
            "checks": {},
        },
        "SpectralBiclustering": {
            "constructor": SpectralBiclustering,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
            },
            "checks": {},
        },
        "SpectralCoclustering": {
            "constructor": SpectralCoclustering,
            "args": lambda ns: {
                **({} if ns.skl_n_clusters is None else {"n_clusters": ns.skl_n_clusters}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
            },
            "checks": {},
        },

        # ---- decision trees ----
        "DecisionTreeClassifier": {
            "constructor": DecisionTreeClassifier,
            "args": lambda ns: {
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "DecisionTreeRegressor": {
            "constructor": DecisionTreeRegressor,
            "args": lambda ns: {
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "ExtraTreeClassifier": {
            "constructor": ExtraTreeClassifier,
            "args": lambda ns: {
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "ExtraTreeRegressor": {
            "constructor": ExtraTreeRegressor,
            "args": lambda ns: {
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },

        # ---- ensemble methods ----
        "AdaBoostClassifier": {
            "constructor": AdaBoostClassifier,
            "args": lambda ns: {
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "AdaBoostRegressor": {
            "constructor": AdaBoostRegressor,
            "args": lambda ns: {
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "BaggingClassifier": {
            "constructor": BaggingClassifier,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_max_samples is None else {"max_samples": ns.skl_max_samples}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "BaggingRegressor": {
            "constructor": BaggingRegressor,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_max_samples is None else {"max_samples": ns.skl_max_samples}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "ExtraTreesClassifier": {
            "constructor": ExtraTreesClassifier,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "ExtraTreesRegressor": {
            "constructor": ExtraTreesRegressor,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "GradientBoostingClassifier": {
            "constructor": GradientBoostingClassifier,
            "args": lambda ns: {
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_subsample is None else {"subsample": ns.skl_subsample}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "GradientBoostingRegressor": {
            "constructor": GradientBoostingRegressor,
            "args": lambda ns: {
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_subsample is None else {"subsample": ns.skl_subsample}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "HistGradientBoostingClassifier": {
            "constructor": HistGradientBoostingClassifier,
            "args": lambda ns: {
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "HistGradientBoostingRegressor": {
            "constructor": HistGradientBoostingRegressor,
            "args": lambda ns: {
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "IsolationForest": {
            "constructor": IsolationForest,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_max_samples is None else {"max_samples": ns.skl_max_samples}),
                **({} if ns.skl_contamination is None else {"contamination": ns.skl_contamination}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "RandomForestClassifier": {
            "constructor": RandomForestClassifier,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "RandomForestRegressor": {
            "constructor": RandomForestRegressor,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_criterion is None else {"criterion": ns.skl_criterion}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_min_samples_split is None else {"min_samples_split": ns.skl_min_samples_split}),
                **({} if ns.skl_min_samples_leaf is None else {"min_samples_leaf": ns.skl_min_samples_leaf}),
                **({} if ns.skl_max_features is None else {"max_features": ns.skl_max_features}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "RandomTreesEmbedding": {
            "constructor": RandomTreesEmbedding,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_estimators is None else {"n_estimators": ns.skl_n_estimators}),
                **({} if ns.skl_max_depth is None else {"max_depth": ns.skl_max_depth}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },

        # ---- imputers ----
        "KNNImputer": {
            "constructor": KNNImputer,
            "args": lambda ns: {
                **({} if ns.skl_n_neighbors is None else {"n_neighbors": ns.skl_n_neighbors}),
                **({} if ns.skl_weights is None else {"weights": ns.skl_weights}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
            },
            "checks": {},
        },
        "MissingIndicator": {
            "constructor": MissingIndicator,
            "args": lambda ns: {
                **({} if ns.skl_features is None else {"features": ns.skl_features}),
            },
            "checks": {},
        },
        "SimpleImputer": {
            "constructor": SimpleImputer,
            "args": lambda ns: {
                **({} if ns.skl_strategy is None else {"strategy": ns.skl_strategy}),
                **({} if ns.skl_fill_value is None else {"fill_value": ns.skl_fill_value}),
            },
            "checks": {},
        },

        # ---- manifold methods ----
        # ClassicalMDS is MDS with metric=True (Principal Coordinate Analysis)
        "ClassicalMDS": {
            "constructor": MDS,
            "args": lambda ns: {
                "metric": True,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"eps": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "Isomap": {
            "constructor": Isomap,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_neighbors is None else {"n_neighbors": ns.skl_n_neighbors}),
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_metric is None else {"metric": ns.skl_metric}),
            },
            "checks": {},
        },
        "LocallyLinearEmbedding": {
            "constructor": LocallyLinearEmbedding,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_neighbors is None else {"n_neighbors": ns.skl_n_neighbors}),
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_method is None else {"method": ns.skl_method}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "MDS": {
            "constructor": MDS,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"eps": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "SpectralEmbedding": {
            "constructor": SpectralEmbedding,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "TSNE": {
            "constructor": TSNE,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_perplexity is None else {"perplexity": ns.skl_perplexity}),
                **({} if ns.skl_shrinkage is None else {"learning_rate": ns.skl_shrinkage}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },

        # ---- Gaussian mixture models ----
        "BayesianGaussianMixture": {
            "constructor": BayesianGaussianMixture,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_covariance_type is None else {"covariance_type": ns.skl_covariance_type}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "GaussianMixture": {
            "constructor": GaussianMixture,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_covariance_type is None else {"covariance_type": ns.skl_covariance_type}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_n_init is None else {"n_init": ns.skl_n_init}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },

        # ---- Naive Bayes ----
        "BernoulliNB": {
            "constructor": BernoulliNB,
            "args": lambda ns: {
                **({} if ns.skl_alpha is None else {"alpha": ns.skl_alpha}),
                **({} if ns.skl_fit_prior is None else {"fit_prior": ns.skl_fit_prior}),
            },
            "checks": {},
        },
        "CategoricalNB": {
            "constructor": CategoricalNB,
            "args": lambda ns: {
                **({} if ns.skl_alpha is None else {"alpha": ns.skl_alpha}),
                **({} if ns.skl_fit_prior is None else {"fit_prior": ns.skl_fit_prior}),
            },
            "checks": {},
        },
        "ComplementNB": {
            "constructor": ComplementNB,
            "args": lambda ns: {
                **({} if ns.skl_alpha is None else {"alpha": ns.skl_alpha}),
                **({} if ns.skl_fit_prior is None else {"fit_prior": ns.skl_fit_prior}),
            },
            "checks": {},
        },
        "GaussianNB": {
            "constructor": GaussianNB,
            "args": lambda ns: {
                **({} if ns.skl_var_smoothing is None else {"var_smoothing": ns.skl_var_smoothing}),
            },
            "checks": {},
        },
        "MultinomialNB": {
            "constructor": MultinomialNB,
            "args": lambda ns: {
                **({} if ns.skl_alpha is None else {"alpha": ns.skl_alpha}),
                **({} if ns.skl_fit_prior is None else {"fit_prior": ns.skl_fit_prior}),
            },
            "checks": {},
        },

        # ---- SVMs ----
        "LinearSVC": {
            "constructor": LinearSVC,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_penalty is None else {"penalty": ns.skl_penalty}),
                **({} if ns.skl_loss is None else {"loss": ns.skl_loss}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "LinearSVR": {
            "constructor": LinearSVR,
            "args": lambda ns: {
                "fit_intercept": ns.skl_include_bias,
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_loss is None else {"loss": ns.skl_loss}),
                **({} if ns.skl_epsilon is None else {"epsilon": ns.skl_epsilon}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "NuSVC": {
            "constructor": NuSVC,
            "args": lambda ns: {
                **({} if ns.skl_nu is None else {"nu": ns.skl_nu}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "NuSVR": {
            "constructor": NuSVR,
            "args": lambda ns: {
                **({} if ns.skl_nu is None else {"nu": ns.skl_nu}),
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "OneClassSVM": {
            "constructor": OneClassSVM,
            "args": lambda ns: {
                **({} if ns.skl_nu is None else {"nu": ns.skl_nu}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "SVC": {
            "constructor": SVC,
            "args": lambda ns: {
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },
        "SVR": {
            "constructor": SVR,
            "args": lambda ns: {
                **({} if ns.skl_C is None else {"C": ns.skl_C}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_epsilon is None else {"epsilon": ns.skl_epsilon}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
            },
            "checks": {},
        },

        # ---- decomposition ----
        "FactorAnalysis": {
            "constructor": FactorAnalysis,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "FastICA": {
            "constructor": FastICA,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "IncrementalPCA": {
            "constructor": IncrementalPCA,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_batch_size is None else {"batch_size": ns.skl_batch_size}),
            },
            "checks": {},
        },
        "KernelPCA": {
            "constructor": KernelPCA,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_kernel is None else {"kernel": ns.skl_kernel}),
                **({} if ns.skl_gamma is None else {"gamma": ns.skl_gamma if ns.skl_gamma in ('scale', 'auto') else float(ns.skl_gamma)}),
                **({} if ns.skl_poly_degree is None else {"degree": ns.skl_poly_degree}),
                **({} if ns.skl_coef0 is None else {"coef0": ns.skl_coef0}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "LatentDirichletAllocation": {
            "constructor": LatentDirichletAllocation,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_batch_size is None else {"batch_size": ns.skl_batch_size}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "NMF": {
            "constructor": NMF,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "MiniBatchNMF": {
            "constructor": MiniBatchNMF,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_batch_size is None else {"batch_size": ns.skl_batch_size}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "MiniBatchSparsePCA": {
            "constructor": MiniBatchSparsePCA,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_batch_size is None else {"batch_size": ns.skl_batch_size}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "PCA": {
            "constructor": PCA,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        # Note: SparseCoder requires a 'dictionary' array that must be provided externally
        "SparseCoder": {
            "constructor": SparseCoder,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
            },
            "checks": {},
        },
        "SparsePCA": {
            "constructor": SparsePCA,
            "args": lambda ns: {
                "n_jobs": ns.cpus_per_task,
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_max_iter is None else {"max_iter": ns.skl_max_iter}),
                **({} if ns.skl_tol is None else {"tol": ns.skl_tol}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
        "TruncatedSVD": {
            "constructor": TruncatedSVD,
            "args": lambda ns: {
                **({} if ns.skl_n_components is None else {"n_components": ns.skl_n_components}),
                **({} if ns.skl_random_state is None else {"random_state": ns.skl_random_state}),
            },
            "checks": {},
        },
    }

    Y_PIPELINE_REGISTRY = {
        "TransformOutputRavel": {
            "constructor": TransformOutputRavel,
            "args": lambda ns: {},
            "checks": {},
        },
    }

    def __init__(self, args:ArgumentParser, fbase:str):
        '''
        '''
        self.args = args
        self.fbase = fbase
        self.pipeline = self.build_pipeline(args)
        self.y_pipeline = self.build_y_pipeline(args)

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

        # Apply y pipeline (fit on training, transform on val/test)
        if self.y_pipeline is not None:
            self.y_pipeline.fit(sds.outs_training)
            outs_training   = self.y_pipeline.transform(sds.outs_training)
            outs_validation = (self.y_pipeline.transform(sds.outs_validation)
                               if sds.ins_validation is not None else None)
            outs_testing    = (self.y_pipeline.transform(sds.outs_testing)
                               if sds.ins_testing is not None else None)
        else:
            # No y-pipeline
            outs_training   = sds.outs_training
            outs_validation = sds.outs_validation
            outs_testing    = sds.outs_testing

        # Train the model
        self.pipeline.fit(sds.ins_training, outs_training)

        # Log the results
        results = {}

        # Training
        ev = self.pipeline.score(X=sds.ins_training, y=outs_training)
        results['score_training'] = ev

        # Log the details?
        if self.args.log_training_set:
            print_debug('Training predict', 4, self.args.debug)
            results['ins_training'] = sds.ins_training
            results['outs_training'] = outs_training
            results['predict_training'] = self.pipeline.predict(sds.ins_training)

        # Validation Data Set
        if sds.ins_validation is not None:
            ev = self.pipeline.score(X=sds.ins_validation, y=outs_validation)
            results['score_validation'] = ev

            # Log the details?
            if self.args.log_validation_set:
                print_debug('Validation predict', 4, self.args.debug)
                results['ins_validation'] = sds.ins_validation
                results['outs_validation'] = outs_validation
                results['predict_validation'] = self.pipeline.predict(sds.ins_validation)

        # Testing Data Set
        if sds.ins_testing is not None:
            ev = self.pipeline.score(X=sds.ins_testing, y=outs_testing)
            results['score_testing'] = ev

            # Log the details?
            if self.args.log_testing_set:
                print_debug('Testing predict', 4, self.args.debug)
                results['ins_testing'] = sds.ins_testing
                results['outs_testing'] = outs_testing
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
                model = {'model': self.pipeline,
                         'y_preprocessor_pipeline': self.y_pipeline,
                         }
                pickle.dump(model, fp)


    def build_step(self, name: str, ns: Namespace, registry: dict) -> object:
        '''
        Instantiate a single pipeline element from the given registry.

        Existence checks run first; constructor args are evaluated only afterward,
        at the moment the constructor is called.
        '''

        if name not in registry:
            handle_error('Pipeline element %s does not exist' % name, self.args.verbose)

        entry = registry[name]

        missing = {a for a in entry.get("checks", set()) if (not hasattr(ns, a)) or (getattr(ns, a) is None)}
        if missing:
            handle_error(f"pipeline element {name!r}: missing argument(s) {sorted(missing)}", self.args.verbose)

        kwargs = entry["args"](ns)
        return entry["constructor"](**kwargs)

    def build_pipeline(self, ns: Namespace) -> Pipeline:
        '''
        Build the X Pipeline from ns.skl_pipeline.
        '''
        return Pipeline([(name, self.build_step(name, ns, SklearnModeler.PIPELINE_REGISTRY))
                         for name in ns.skl_pipeline])

    def build_y_pipeline(self, ns: Namespace):
        '''
        Build the y Pipeline from ns.skl_y_pipeline, or return None if unset.
        '''
        if not hasattr(ns, 'skl_y_pipeline') or ns.skl_y_pipeline is None:
            return None
        return Pipeline([(name, self.build_step(name, ns, SklearnModeler.Y_PIPELINE_REGISTRY))
                         for name in ns.skl_y_pipeline])
