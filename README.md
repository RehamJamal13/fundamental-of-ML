# Neural Network Implementation

Neural networks are mathematical models inspired by the brain's structure and learning algorithms. They are used to process complex and non-linearly separable data, making predictions by learning from examples. This repository provides a simple implementation of a neural network and the backpropagation algorithm.

![Neural Network](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/33518920-ed38-4cea-acab-7918068f6c32)

## Neural Network Architecture

Neural networks consist of interconnected nodes called neurons, organized in layers. Each neuron receives input, performs computations, and produces an output. The architecture includes:
- Input Layer
- Hidden Layers
- Output Layer

![Feed Forward Neural Network](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/3169222c-b4ed-45a3-8fcd-9715152c51cc)

## Forward Pass

The forward pass involves computing the output given an input:
1. **Input Layer**: Data is fed into the network.
2. **Hidden Layers**: Inputs pass through hidden layers, where each neuron computes a weighted sum and applies the activation function (e.g., sigmoid).
3. **Output Layer**: The output is produced.

## Backpropagation

Backpropagation is an algorithm used to train neural networks by propagating errors backward through the network and adjusting weights to minimize the difference between actual and desired outputs.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by adjusting network parameters iteratively.

## Activation Function: Sigmoid

The sigmoid function squashes input values between 0 and 1, making it suitable for binary classification problems.

![Sigmoid Function](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/3a256fbe-8f10-44aa-9475-39d635c0d4a1)

## Loss Function: Cross-Entropy

Cross-entropy loss measures the difference between predicted and true label distributions, often used in classification problems.

![Cross-Entropy Loss](https://github.com/RehamJamal13/Machine-Learning-Algorithms/assets/102676168/9a9f3180-84c3-4eb5-a4f2-08f40e5adc08)

## Weight Update

The weight update step adjusts weights based on computed gradients using the learning rate.
Weight Update:
# Weight Update
# 1. W1 Update: Adjust the weights of the first layer using the gradient and the learning rate.
W1 = W1 - alpha * dW1

# 2. W2 Update: Adjust the weights of the second layer using the gradient and the learning rate.
W2 = W2 - alpha * dW2

# 3. b1 Update: Adjust the biases of the first layer using the gradient and the learning rate.
b1 = b1 - alpha * db1

# 4. b2 Update: Adjust the biases of the second layer using the gradient and the learning rate.
b2 = b2 - alpha * db2



## Usage

To use the neural network implementation, follow the instructions in the repository's documentation.
















# Principal Component Analysis (PCA) Implementation

Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in machine learning and data analysis. It helps in reducing the number of features in a dataset while preserving the essential information. In this section, we'll discuss the implementation of PCA using Python, both with and without the use of the `sklearn` library.
![download](https://github.com/RehamJamal13/fundamental-of-ML/assets/102676168/d6971be9-8f0d-4457-adb2-2ab88df471a4)

## Introduction to PCA

PCA is a statistical method that transforms high-dimensional data into a lower-dimensional form by identifying the principal components of variation in the data. These principal components are orthogonal vectors that capture the maximum variance in the dataset.

## Implementation Steps

### 1. Standardization

Before applying PCA, it's essential to standardize the data by subtracting the mean and dividing by the standard deviation of each feature. This step ensures that all features have a similar scale.

### 2. Computing the Covariance Matrix

Next, we compute the covariance matrix of the standardized data. The covariance matrix provides information about the relationships between different features in the dataset.

### 3. Eigendecomposition

We then perform eigendecomposition on the covariance matrix to find its eigenvalues and eigenvectors. These eigenvalues represent the amount of variance captured by each principal component, while the eigenvectors represent the directions of maximum variance.

### 4. Choosing the Number of Components

We choose the number of principal components based on various criteria, such as the explained variance or Kaiser's rule.

### 5. Projecting the Data

Finally, we project the original data onto the subspace spanned by the selected principal components to obtain the lower-dimensional representation of the dataset.

## PCA Implementation with Python

We provide a Python implementation of PCA using both manual coding and the `sklearn` library. The manual implementation covers the essential steps of PCA, including standardization, covariance matrix computation, eigendecomposition, and data projection. On the other hand, the `sklearn` implementation offers a convenient way to perform PCA with minimal code.

### Manual PCA Implementation

We define a custom PCA class that encapsulates the essential PCA methods, such as fitting the model to data and transforming the data onto the principal components subspace. The class implements standardization, covariance matrix computation, eigendecomposition, and data projection functionalities.

### PCA with sklearn

We also demonstrate how to perform PCA using the `sklearn.decomposition.PCA` class. This class provides a high-level interface for performing PCA, allowing us to fit the model to data and transform the data in just a few lines of code.

## Conclusion

PCA is a powerful technique for dimensionality reduction and data visualization. By identifying the principal components of variation in a dataset, PCA enables us to extract meaningful insights and reduce computational complexity. Whether implemented manually or using libraries like `sklearn`, PCA remains a valuable tool in the data scientist's toolkit.


