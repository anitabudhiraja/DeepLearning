# Deep Learning Fundamentals & Practical Implementations

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat&logo=Keras&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

A comprehensive repository containing advanced deep learning experiments, practical implementations, and theoretical foundations. This collection spans across core neural network concepts, Computer Vision (CV), Natural Language Processing (NLP), and predictive modeling using structured data. 

This repository is designed to bridge the gap between theoretical architectures and practical engineering, providing robust implementations for research and development.

---

## 📑 Table of Contents
1. [Overview](#-overview)
2. [Repository Structure](#-repository-structure)
   - [Core Deep Learning Concepts](#1-core-deep-learning-concepts--optimization)
   - [Computer Vision (CV)](#2-computer-vision-cv)
   - [Natural Language Processing (NLP)](#3-natural-language-processing-nlp)
   - [Predictive Modeling & Tabular Data](#4-predictive-modeling--tabular-data)
   - [Advanced Architectures & Techniques](#5-advanced-architectures--techniques)
3. [Theory & Documentation](#-theory--documentation)
4. [Getting Started](#%EF%B8%8F-getting-started)
5. [Prerequisites](#-prerequisites)

---

## 🔬 Overview
This repository serves as a centralized hub for executing deep learning workflows. It includes side-by-side performance comparisons (e.g., ANN vs. CNN, CPU vs. GPU) and tackles real-world data challenges such as class imbalance and data augmentation. Implementations are primarily built using **TensorFlow/Keras** and presented via **Jupyter Notebooks** for reproducible research and experimentation.

---

## 🗂 Repository Structure

### 1. Core Deep Learning Concepts & Optimization
Foundational mechanics of training neural networks, exploring loss functions, and hardware utilization.
* `categorical_crossentropy.ipynb` & `BinaryCrossEntropy.ipynb`: In-depth mathematical and practical exploration of classification loss functions.
* `practical_7_activation_functions.ipynb`: Analysis of different activation functions (ReLU, Sigmoid, Tanh, etc.) and their impact on vanishing/exploding gradients.
* `practical_8_deep_ann.ipynb`: Building and scaling deep Artificial Neural Networks.
* `practical_11_cpu_vs_gpu.ipynb`: Computational benchmarking comparing training times and resource allocation between CPU and GPU hardware.

### 2. Computer Vision (CV)
Image processing, feature extraction, and image classification architectures.
* **MNIST**: 
  * `MNIST.ipynb`, `practical_1_mnist.ipynb`, `practical_2_mnist_classification.ipynb`
  * `ANN_vs_CNN_MNIST.ipynb`: An architectural comparison between dense networks and convolutional networks on the MNIST dataset.
* **CIFAR-10 & ImageNet**:
  * `CNN_Cifar10.ipynb`: Multi-class image classification using Convolutional Neural Networks.
  * `Classify_Images_CNN_Imagenet.ipynb`: Leveraging pre-trained ImageNet weights for transfer learning and feature extraction.
* **Cats vs. Dogs**: 
  * `Image_cats_dogs/cats_dogs.ipynb`: Binary image classification demonstrating spatial hierarchies and basic computer vision pipelines.

### 3. Natural Language Processing (NLP)
Sequence modeling and text classification focusing on sarcasm detection.
* `NLP_tokenizer.ipynb`: Text preprocessing, tokenization, and padding sequences for neural network ingestion.
* `Sarcasm_Using_RNN.ipynb`: Basic Recurrent Neural Networks for sequence classification.
* `LSTM_RNN_Sarcasm.ipynb`: Advanced sequence modeling utilizing Long Short-Term Memory (LSTM) networks to handle long-range dependencies in text.
* `Sarcasm_CNN.ipynb`: Utilizing 1D Convolutions for rapid feature extraction from text sequences.

### 4. Predictive Modeling & Tabular Data
Applying deep learning to structured, tabular datasets for classification and regression.
* **Loan Prediction Project**: 
  * `Solving_Loan_Prediction_problem_using_Neural_Network.ipynb`: End-to-end pipeline handling missing values, categorical encoding, and neural network training to predict loan approval.
* **Healthcare / Biology**:
  * `Diabetes_prediction/diabetes_deeplearning.ipynb`: Building predictive models on healthcare data (`diabetes_dl.csv`).
  * `practical_3_disease_prediction.ipynb`: General disease prediction using structured medical parameters.
  * `DNAClassification/DNA Classification.ipynb`: Sequence mapping and classification of genomic data.

### 5. Advanced Architectures & Techniques
Tackling real-world machine learning challenges and utilizing generative/forecasting models.
* `practical_5_imbalanced_data.ipynb`: Strategies for training models on skewed datasets (e.g., class weights, SMOTE, focal loss).
* `practical_6_data_augmentation.ipynb`: Artificially expanding training datasets to prevent overfitting in vision models using `ImageDataGenerator`.
* `practical_9_lstm_forecasting.ipynb`: Time-series forecasting utilizing LSTM architectures.
* `practical_10_autoencoder.ipynb`: Unsupervised learning, dimensionality reduction, and anomaly detection using Autoencoders.

---

## 📚 Theory & Documentation
Alongside the executable code, the repository contains extensive theoretical documentation to support academic instruction and engineering reviews:
* **PDF Guides**: `DL_1.pdf`, `DL_2.pdf`, `DL_3.pdf`, `DL_4.pdf` provide comprehensive notes on neural network mathematics, backpropagation, and network topologies.
* **HTML Modules**: `theory_practical1.html` through `theory_practical11.html` offer web-accessible documentation directly correlated to the respective Jupyter notebooks.

---

## ⚙️ Getting Started

### Clone the Repository
```bash
git clone [https://github.com/your-username/deeplearning.git](https://github.com/your-username/deeplearning.git)
cd deeplearning
```

### Launch the Environment
Navigate to the specific module directory or launch Jupyter from the root to access all files:
```bash
jupyter notebook
```

---

## 🛠 Prerequisites

Ensure you have a Python 3.8+ environment set up. The primary dependencies required to execute the notebooks in this repository are:

* `tensorflow` (>= 2.x)
* `keras`
* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `seaborn` (for confusion matrices and data visualization)
* `nltk` / `spacy` (for NLP preprocessing tasks)

You can install the standard deep learning stack via pip:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn jupyter
```

*(Note: For GPU acceleration in `practical_11_cpu_vs_gpu.ipynb`, ensure you have the appropriate CUDA Toolkit and cuDNN libraries configured for your hardware).*

---
*Maintained for research, engineering excellence, and the advancement of computational sciences.*
