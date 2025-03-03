Neural Network

Satwika Kallem

Student Id: 700771340

Crn: 23848

Deep Learning with Cloud Computing, CNNs, and ResNet

Overview

This repository contains implementations and comparisons related to deep learning concepts using cloud computing platforms, convolution operations, feature extraction, and CNN architectures such as AlexNet and ResNet. The following tasks are covered:

Cloud Computing for Deep Learning

Definitions of elasticity and scalability.

Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio.

Convolution Operations

Implementing convolution with different strides and paddings using NumPy and TensorFlow/Keras.

Printing feature maps for various configurations.

CNN Feature Extraction

Edge detection using Sobel filters with OpenCV.

Implementation of Max Pooling and Average Pooling with TensorFlow/Keras.

CNN Architectures Implementation

Implementation of a simplified AlexNet model in TensorFlow/Keras.

Implementation of a Residual Block and a simple ResNet-like model.

Prerequisites

Ensure you have the following dependencies installed:

pip install numpy tensorflow keras opencv-python

Folder Structure

/DeepLearning
│── cloud_computing.md  # Explanation of elasticity, scalability, and platform comparison
│── convolution.py       # Convolution operations with different stride and padding
│── feature_extraction.py # Edge detection with Sobel filters and pooling operations
│── alexnet.py           # Implementation of AlexNet model
│── resnet.py            # Implementation of a simple ResNet-like model
│── README.md            # Project documentation

Usage

1. Cloud Computing for Deep Learning

Refer to cloud_computing.md for theoretical explanations and comparisons.

2. Convolution Operations

Run the convolution.py script to generate feature maps for different stride and padding configurations:

python convolution.py

3. CNN Feature Extraction

Run feature_extraction.py to perform edge detection using Sobel filters and demonstrate pooling operations:

python feature_extraction.py

4. CNN Architectures

To execute the AlexNet model implementation:

python alexnet.py

To execute the ResNet-like model implementation:

python resnet.py

Expected Outputs

Convolution Operations: Feature maps for different stride and padding configurations.

Edge Detection: Displays original image, Sobel-X, and Sobel-Y results.

Pooling Operations: Prints original, max-pooled, and average-pooled matrices.

CNN Models: Prints model summaries for AlexNet and ResNet.

License

This project is open-source and licensed under the MIT License.