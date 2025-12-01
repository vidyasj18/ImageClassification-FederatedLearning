# Image Classification using Federated Learning
This Task implements an Image Classification model using Federated Learning (FL) on the CIFAR-10 dataset. And the system uses Flower (FLWR) for the federated server–client 
communication and TensorFlow/Keras for model training.

## Techstack and Libraries used
- Tensorflow : For model training and neural networks.
- Flower : For building server-client communication in federated learning.
- Numpy :  For numerical operations and data manipulation.

## Dataset used 
The dataset used in the task is CIFAR-10. https://www.cs.toronto.edu/%7Ekriz/cifar.html

CIFAR-10 dataset built into TensorFlow contains:
- 60,000 images
- 10 classes
- 32×32 RGB images
Clients do not download separately — TF loads automatically.

## Built this model using:
- 1 server
- 2 clients
- CIFAR-10 dataset
- Federated Averaging (FedAvg) strategy

## Key Features
- Each client trains on its own local data partition
- The server aggregates weights using FedAvg
- A global model is created and evaluated
- Model saved as global_model/model.h5
- Federated metrics logged and displayed

## Installation steps
- Clone this repository
- Install dependencies

```
pip install flwr tensorflow
```

## Model Architecture
This project uses a simple Convolutional Neural Network (CNN) designed for the CIFAR-10 dataset.

- 2 Convolution layers - Applies filters of size 3×3. And detects low-level features like edges, textures, corners
- MaxPooling - Reduces image size by taking the maximum value in a region (usually 2×2).
- Flatten - Converts the 2D feature maps from convolution layers into a 1D feature vector.
- Dense → ReLU - A fully connected layer with 128 neurons. Learns high-level combinations of features extracted earlier.
- Dense → Softmax (10 classes) - Produces 10 probabilities (one for each CIFAR-10 class). Softmax ensures probabilities sum to 1.

#### Compiled with:
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

