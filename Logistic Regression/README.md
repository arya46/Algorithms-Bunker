# Logistic Regression Playground

## Overview
This script contains the implementation of Logistic Regression Algorithm from scratch in Python.

## What is Logistic Regression?
Logistic Regression is one of the most common machine learning algorithms for classification. It a statistical model that uses a logistic function to model a binary dependent variable. 


## How it works?

Please refer to my [Medium blog](https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59) to learn about LR in greater details.

You can play around with the sandbox version of this algorithm [here!](https://play-with-lr.herokuapp.com/)

## Parameters: 
The model accepts the following params:
- `learning_rate` : The tuning parameter for the optimization algorithm (here, Gradient Descent) that determines the step size at each iteration while moving toward a minimum of the cost function.
- `max_iter` : Maximum number of iterations taken for the optimization algorithm to converge
- `penalty` : None or 'l2', Option to perform L2 regularization.
- `C` : Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization. 
- `tolerance` : Value indicating the weight change between epochs in which gradient descent should terminated. 

## Methods:
The model has the following exposed methods:
- `predict()` : Predicts class labels for an observation
- `predict_proba()` : Predicts the estimate for a class
- `get_params()` : Returns the coefficients and intercept

## Usage
The API's for my implementation has been made similar to sklearn API's for easier use:
```
from logreg_classifier import LogisticRegression

# fit the data
clf = LogisticRegression()
clf.fit(X,y)

# predict probabilities
probs = clf.predict_proba(x_test)

# predict class labels
preds = clf.predict(x_test)
```

__Results on a dummy dataset__:

<img src="https://miro.medium.com/max/518/1*cg5u-0iKthH82o0d7yu8uA.jpeg">
