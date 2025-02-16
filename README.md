Neural Network 

Satwika Kallem 

Student Id: 700771340

Crn: 23848 

TensorFlow Neural Network Training with TensorBoard Logging

Overview

This project demonstrates various TensorFlow operations, including tensor manipulations, loss function comparisons, optimizer performance evaluations, and logging training metrics using TensorBoard.

Requirements

Python 3.x

TensorFlow

NumPy

Matplotlib

1. Tensor Manipulations & Reshaping

Description

This code creates and manipulates tensors using TensorFlow, demonstrating reshaping, transposing, and broadcasting operations.

Key Steps

Create a random tensor of shape (4,6) and determine its rank and shape.

Reshape the tensor into (2,3,4) and transpose it to (3,2,4).

Perform broadcasting of a smaller tensor (1,4) to match a larger tensor and add them.

Print the rank and shape before and after transformations.

2. Loss Functions & Hyperparameter Tuning

Description

This code calculates and compares different loss functions (Mean Squared Error and Categorical Cross-Entropy) and evaluates how minor changes in predictions affect the loss values.

Key Steps

Define true labels and predicted values.

Compute MSE and CCE loss values.

Modify predictions slightly and observe loss variations.

Plot a bar chart comparing MSE and Cross-Entropy loss values using Matplotlib.

3. Train a Model with Different Optimizers

Description

This code trains a simple neural network on the MNIST dataset using two different optimizers (Adam and SGD) and compares their performance.

Key Steps

Load the MNIST dataset and normalize it.

Define a feedforward neural network with one hidden layer of 128 neurons.

Train the model separately using Adam and SGD optimizers.

Plot a graph comparing the validation accuracy of both optimizers.

4. Train a Neural Network and Log to TensorBoard

Description

This code trains a neural network on the MNIST dataset while logging training metrics to TensorBoard for visualization.

Key Steps

Load and preprocess the MNIST dataset.

Define a simple neural network architecture.

Set up TensorBoard logging in the "logs/fit/" directory.

Train models with Adam and SGD optimizers and log the results.

Plot accuracy comparisons between the optimizers.

Launch TensorBoard with:

tensorboard --logdir=logs/fit

Expected Outcomes

Tensor manipulations showcase fundamental TensorFlow operations.

Loss function variations provide insights into model evaluation.

Optimizer comparisons highlight the differences between Adam and SGD.

TensorBoard logging enables visualization of training accuracy and loss trends.

License

This project is open-source and available for use under the MIT License.
