# Deep Learning Assignment - Autoencoders and RNNs

## Student Information

* **Name:** Satwika Kallem
* **Student ID:** 700771340
* **Course:** Neural Network

## Overview

This repository contains the source code for a deep learning assignment focused on implementing autoencoders and recurrent neural networks (RNNs). The assignment consists of four tasks:

1.  **Basic Autoencoder:** Implementing a fully connected autoencoder for image reconstruction using the MNIST dataset.
2.  **Denoising Autoencoder:** Modifying the basic autoencoder to denoise images by adding Gaussian noise.
3.  **RNN for Text Generation:** Training an LSTM-based RNN to generate text using a character-level model.
4.  **Sentiment Classification using RNN:** Training an LSTM-based sentiment classifier using the IMDB dataset.

## Repository Contents

* `Q1_basic_autoencoder.py`: Python script for implementing the basic autoencoder.
* `Q2_denoising_autoencoder.py`: Python script for implementing the denoising autoencoder.
* `Q3_rnn_text_generation.py`: Python script for implementing the RNN text generator.
* `Q4_sentiment_classification.py`: Python script for implementing the sentiment classifier.
* `README.md`: This file, explaining the assignment and repository contents.
* `shakespeare.txt`: Text file containing the Shakespeare sonnets used for Q3.

## Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```
2.  **Install Dependencies:**
    Ensure you have Python 3.x and the following libraries installed:
    ```bash
    pip install tensorflow numpy scikit-learn matplotlib seaborn
    ```
3.  **Run the Scripts:**
    Execute each Python script to train and evaluate the corresponding models:
    ```bash
    python Q1_basic_autoencoder.py
    python Q2_denoising_autoencoder.py
    python Q3_rnn_text_generation.py
    python Q4_sentiment_classification.py
    ```
4.  **View Results:**
    The scripts will output the results, including reconstructed images, generated text, and classification reports. Visualizations will be displayed in pop-up windows.

## Code Explanation

### Q1: Basic Autoencoder

* Loads the MNIST dataset.
* Defines a fully connected autoencoder with an encoder (784 -> 32) and a decoder (32 -> 784).
* Trains the model using binary cross-entropy loss.
* Plots the original and reconstructed images.
* Analyzes the effect of latent dimension size (16, 64) on reconstruction quality.

### Q2: Denoising Autoencoder

* Modifies the basic autoencoder by adding Gaussian noise to input images.
* Trains the model to reconstruct clean images from noisy inputs.
* Compares the performance of the basic and denoising autoencoders.
* Explains a real-world application (medical imaging).

### Q3: RNN for Text Generation

* Loads the "Shakespeare Sonnets" dataset.
* Converts text into sequences of characters using one-hot encoding.
* Defines an LSTM-based RNN to predict the next character.
* Trains the model and generates new text using temperature scaling.
* Explains the effect of temperature on text generation randomness.

### Q4: Sentiment Classification Using RNN

* Loads the IMDB sentiment dataset.
* Preprocesses the text data by tokenization and padding sequences.
* Trains an LSTM-based sentiment classifier.
* Generates a confusion matrix and classification report.
* Explains the importance of the precision-recall tradeoff in sentiment classification.

