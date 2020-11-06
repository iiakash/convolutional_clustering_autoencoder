# Deep Convolutional Clustering Autoencoder

## The reopository contains deep convolutional clustering autoencoder method implementation with PyTorch

## Overview

The application of technologies like Internet of Things(IoT) have paved the way to solve complex industrial problems with the help of large amounts of information. Industrial systems are now intertwined with each other constantly sharing data among them. Vast quantities of this data is unlabeled, and it is a mammoth task to label the data manually. If there are anomalies present in the data it is even harder to find and label the anomalous samples. The objective of unsupervised anomaly detection solves this problem by partitioning the input data into distinct normal and anomalous regions. In this project, an unsupervised anomaly detection approach by jointly using a Convolutional Autoencoder, and  the K-means clustering algorithm is implemented. The encoder part of the autoencoder is used to map a low dimensional feature representation of the original data. The decoder is responsible for reconstructing the encoded data to the original form. The convolutional autoencoder is optimized using the error produced from the reconstructed data. A clustering algorithm (i.e., K-Means) is simultaneously being applied on the latent feature representation to initialize two cluster centers. This allows us to jointly optimize the network by combining an additional clustering loss. The clustering loss enforces the encoder to learn clustering friendly representation by minimizing the mean-squared error between the cluster centers and the data samples.

## Data

The first dataset used for this project is provided by Bültmann GmbH. The dataset contains high frequency sensor data from a metal peeling machine. Due to the sensitive nature of the data used in this project, the data is not provided in this repository. The data is considered exclusive for this project and can not be used for any other purpose.

The second dataset used for this project is from the Tennessee Eastman Process. The Tennessee Eastman Process is a simulation of a chemical plant which is based on an actual process in the Eastman Chemical Company located in Tennessee, USA. The Tennessee Eastman process was originally created by Downs and Vogel as a process control challenge problem. 

The generated dataset from the Tennessee Eastman Process consists of 22 continuous process measurements, 19 component analysis measurements, and 12 manipulated variables. The dataset consists of 21 pre-programmed faults, among which 16 are known fault cases, and 5 fault cases are unknown. Both the training and testing datasets include a total of 52 observed variables. The training dataset consists of 22 different simulation runs, and simulation 0 is fault free. In our case, this simulation is considered as our normal data samples. Simulations 1 to 21 were generated for 21 fault cases, and in our case all of these 21 simulations are considered as anomalous data samples. Similarly, the testing data set contains 22 different simulations, the first one being the normal case, and the rest are simulations for different fault cases. All of the 22 data sets have 960 observations each, and 52 observed variables.

## Dependencies

| Framework | Version |
|-----------|---------|
| Numpy     | 1.18.5  |
| Pandas    | 1.0.4   |
| PyTorch   | 1.4.0   |
| Sklearn   | 0.24    |
| Matplotlib| 3.2.1   |

The code was written and tested on Python 3.7
