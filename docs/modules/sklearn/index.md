---
title: Scikit-Learn Models
nav_order: 50
parent: Zero2Neuro Modules
has_children: true
---

# Scikit-Learn Models
As an alternative to deep neural networks, one can also construct
[Scikit-Learn](https://scikit-learn.org/stable/) based models.  Using
the arguments, the user specifies a [Scikit-Learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), which
consists of a sequence of individual Scikit-Learn modules.  A pipeline
typically consists of some number of preprocessing modules (e.g., feature
normalization, or imputation), followed by a standard supervised module
(e.g., linear regression, decision tree regression), or
unsupervised model (e.g., k-means clustering).  


## Example 1: Linear Regression

[Linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) can be performed using the training inputs and
corresponding outputs.  This can be configured using a network or
model file:
```
--network_type=sklearn

--skl_pipeline
LinearRegression
```

## Example 2: Linear Regression with Polynomial Features

We can first transform the input features using a [Polynomial
transformation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html),
and then perform linear regression using this expanded feature space.
The model file can look like this:


```	
--network_type=sklearn

--skl_pipeline
PolynomialFeatures
LinearRegression

--skl_poly_degree=2
```

## Example 3: Ridge Regression with Polynomial Features

[Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) is a linear regression method that adds a
regularization term that helps to prevent the model from overfitting
the input data.  Here is an example model file:


```	
--network_type=sklearn

--skl_pipeline
PolynomialFeatures
Ridge

--skl_poly_degree=2
--L2_regularization=0.01
```

## Example 4: Decision Tree Classifier

Using a single decision tree to solve the problem


```	
--network_type=sklearn
--skl_pipeline
DecisionTreeClassifier

```

## Example 5: Random Forest Classifier

Here, we are using a forest of decision trees to solve the problem.  Each tree can have a depth of at most 2.

Note: the _RandomForestClassifier_ expects the desired outputs to be a vector (the base representation for tabular data uses a Nx1 matrix).  _TransformOutputRavel_ transforms the desired outputs so that they are in a format that is acceptable to this classifier.

```
--network_type=sklearn

--skl_y_pipeline
TransformOutputRavel
--skl_pipeline
RandomForestClassifier

--skl_n_estimators=100
--skl_max_depth=2
```


## Details
- All pipeline elements in Zero2Neuro use the same name as the corresponding Scikit-Learn class constructors (e.g., LinearRegression, Ridge, PolynomialFeatures). 
- ```--save_model``` will save the trained Pipeline model into a
pickle file named XXX_model.pkl.  
   - The pickle file contains a single dictionary.  
   - The dictionary contains a single key ('model'), whose value is the model.
- Following training, a results file will be created.  This file is
named XXX_results.pkl and contains many of the same keys as the deep
neural network results files, including:
   - scores for the training, validation, and testing data sets
   - the inputs, outputs, and predictiosn for the training
(```--log_training_set```), validation (```--log_validation_set```), and
testing (```--log_testing_set```) data sets.

## Supported Scikit-Learn Modules

The following types of Scikit-Learn modules are supported in
Zero2Neuro:

- [Preprocessors](preprocessors.md): feature scaling, polynomial features, and spline transformations
- [Linear Models](linear_models.md): regression, classification, and regularized linear methods
- [Clustering](clustering.md): unsupervised partitioning and density-based methods
- [Decision Trees](decision_trees.md): single-tree classifiers and regressors
- [Decomposition Methods](decomposition.md): matrix factorization and dimensionality reduction (PCA, NMF, ICA, SVD)
- [Ensemble Methods](ensembles.md): boosting, bagging, and random forest methods
- [Imputers](imputers.md): missing value imputation
- [Gaussian Mixture Models](gaussian_mixture.md): probabilistic density estimation
- [Manifold Methods](manifold.md): non-linear dimensionality reduction
- [Naive Bayes](naive_bayes.md): probabilistic classifiers based on Bayes' theorem
- [Support Vector Machines](svms.md): kernel-based classification and regression

- [Desired Output Preprocessing](output_preprocessing.md): a separate Pipeline for implementing transformations of the desired outputs.



## Scikit-Learn Notes
- Scikit-Learn models can only be trained and evaluated using data
that are **not** in the TF-Dataset format.  
- There is currently no compatibility checking to ensure that the sequence of Scikit-Learn models in the specified pipeline will function together.

