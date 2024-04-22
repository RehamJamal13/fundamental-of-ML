# Logistic Regression Implementation

This repository contains a simple implementation of logistic regression in Python, along with a README file to explain the concept and usage.

## Logistic Regression

Logistic regression is a statistical method used for binary classification problems. It models the probability that a given input belongs to a particular class using the logistic function (sigmoid function).

## Implementation Details

### `accuracy` Function
- Calculates the accuracy of the model's predictions.

### `cross_entropy_loss` Function
- Calculates the cross-entropy loss between true and predicted values.

### `train_test_split` Function
- Splits the dataset into training and testing sets.

### `LogisticRegression` Class
- Implements the logistic regression model.
- Methods:
  - `add_ones`: Adds a column of ones to the input features.
  - `sigmoid`: Computes the sigmoid function.
  - `predict_proba`: Computes the probabilities of the input belonging to the positive class.
  - `predict`: Predicts the class labels based on probabilities.
  - `fit`: Fits the model to the training data using gradient descent.
