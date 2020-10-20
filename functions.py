import numpy as np
from numpy import array
import pandas as pd

import torch
import torch.nn as nn

from collections import namedtuple
from itertools import product

import time

from collections import OrderedDict

from sklearn.cluster import KMeans
import sklearn.metrics

import os

import matplotlib.pyplot as plt

def downsample(arr, downsampling_step):

    """downsamples the data by means of averaging over the number of specified steps"""

    end =  downsampling_step * int(len(arr)/downsampling_step)
    return np.mean(arr[:end].reshape(-1, downsampling_step), 1)


def split_time_series(time_series, sequence_length):

    """splits the univariate time series data into specified number of sequences"""

    X = list()
    for i in range(len(time_series)):
        #find the end of the pattern
        end_ix = i + sequence_length
        #break the loop if we are beyond the pattern
        if end_ix > len(time_series)-1:
            break
        #gather the windowed parts of the time series
        seq_x = time_series[i:end_ix]
        X.append(seq_x)
    return array(X)


def remove_zeros(np_array):

    """removes all zero values from a given array"""

    return np.delete(np_array, np.argwhere(np_array==0))


def divide_batches(array, n):

    """returns an array in n number of batches"""

    for i in range(0, len(array), n):
        yield array[i:i+n]


def timer(start, end):

    """returns hours, minutes, seconds"""

    hours, remainder = divmod(end-start, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


def kmeans(learned_filter, n_clusters, n_init = 10):

    """returns labels, centers and assignment of the centers.
       slower computation than kmeansalter"""

    cluster_ass = []
    km = KMeans(n_clusters = n_clusters, init = 'k-means++',
                n_init = n_init, max_iter = 300, random_state = 0).fit(learned_filter)
    cluster_labels = km.labels_
    cluster_centers = torch.from_numpy(km.cluster_centers_)
    for i in range(len(cluster_labels)):
        assignment = cluster_labels[i].item()
        cluster_ass.append(cluster_centers[assignment,:].unsqueeze(0))
        assigned_center = torch.cat(cluster_ass)


    return cluster_labels, cluster_centers, assigned_center


def kmeansalter(learned_filter, n_clusters, n_init = 10):

    """returns labels, centers and assignment of the centers to each data point.
    faster algorithm than kmeans"""

    center_assignment = np.empty((len(learned_filter), learned_filter.shape[1]))
    km = KMeans(n_clusters = n_clusters, init = 'k-means++',
                n_init = n_init, max_iter = 300, random_state = 0).fit(learned_filter)
    cluster_labels = km.labels_
    cluster_centers = km.cluster_centers_
    zeros_idx = np.where(cluster_labels == 0)
    ones_idx = np.where(cluster_labels == 1)
    for i in zeros_idx[0]:
        center_assignment[i] = cluster_centers[0]
    for i in ones_idx[0]:
        center_assignment[i] = cluster_centers[1]

    assigned_center = torch.from_numpy(center_assignment)


    return cluster_labels, cluster_centers, assigned_center.float()


def rank_channels_alter(centers):

    """ranks the clustering freindly channels according to the center distance metric.
    the further the centers are apart the better the channels. returns the center distances
    in each channel and also the ranks of the distances"""

    distance_between_centers = []
    no_of_channels = centers.shape[1]
    for i in range(no_of_channels):
        channel_centers = centers[:,i,:]
        dist = np.linalg.norm(channel_centers[0].numpy()-channel_centers[1].numpy())
        distance_between_centers.append(dist)
        center_distances = np.asarray(distance_between_centers)
        temp_center_distances = center_distances.argsort()
        ranks_of_center_distances = np.empty_like(temp_center_distances)
        ranks_of_center_distances[temp_center_distances] = np.arange(len(center_distances))


    return  center_distances, ranks_of_center_distances

def rank_channels(centers):

    """subject to question. needs debugging"""

    distance_between_centers = []
    no_of_channels = centers.shape[1]
    for i in range(no_of_channels):
        channel_centers = centers[:,i,:]
        dist = np.linalg.norm(channel_centers[0]-channel_centers[1])
        distance_between_centers.append(dist)
        center_distances = np.asarray(distance_between_centers)
        ranks_of_center_distances = np.argsort(center_distances)


    return  center_distances, ranks_of_center_distances

def consecutive(data, stepsize=1):

    """find the normal and anomalous sequences in the labeled dataset"""

    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_bottleneck_name(network):

    """retrieve the name of the bottleneck"""

    layer_names = []

    for name, parameters in network.named_parameters():
        if name[0] == 'c' and name[-1] == 't':

            layer_names.append(name)


    return layer_names[len(layer_names)-1]

#function: print_weights

class network_reporter():

    """printing some network parameters"""


    def print_weights_grads(network):

        for name, params in network.named_parameters():
            print('Layer: ', name, 'Weight: ', params)


    def print_grads(network):

        for name, params in network.named_parameters():
            print('Layer: ', name, 'Grad: ', params.grad)


    def print_network(network):

        print('Network Architecture: \n', network)


class cm_accuracy:

    """the metrics class. includes normalized mutual inforomation score, adjusted rand index
    and accuracy. accuracy is derived from confusion matrix.

    accuracy = (true_positive + true negative)/(true_positive + true_negative + false_positive + false negative)"""

    #nmi = sklearn.metrics.normalized_mutual_info_score
    #ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def accuracy(confusion_matrix):

        return (confusion_matrix[0][0]+confusion_matrix[1][1])/sum(sum(confusion_matrix))

class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from sklearn.utils.linear_assignment_ import linear_assignment
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / labels_pred.size



def createFolder(directory):

    """creates a folder"""

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


class RMSELoss(torch.nn.Module):

    """root mean squared loss of the pytorch.nn module"""

    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

class Plotter():

    """class to plot the results"""

    def __init__(self, embeddings, list_of_nmi, list_of_ari, list_of_acc, list_of_centers, list_of_ranks_of_center_distances, list_of_losses, top_nmi_maps_idx, top_ari_maps_idx, top_acc_maps_idx):

        self.embeddings = embeddings
        self.list_of_nmi = list_of_nmi
        self.list_of_ari = list_of_ari
        self.list_of_acc = list_of_acc
        self.list_of_centers = list_of_centers
        self.list_of_ranks_of_center_distances = list_of_ranks_of_center_distances
        self.list_of_losses = list_of_losses
        self.top_nmi_maps_idx = top_nmi_maps_idx
        self.top_ari_maps_idx = top_ari_maps_idx
        self.top_acc_maps_idx = top_acc_maps_idx

    def plot_network_loss(self, color = 'blue'):

        network_loss = self.list_of_losses[0]
        figure = plt.figure(figsize = (8,6))
        plt.plot(network_loss, label = 'Network Loss',color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Network Loss')
        plt.title('Plot of Network Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Network Loss.png', dpi = 300)
        plt.show()

    def plot_clustering_loss(self, color = 'green'):

        clustering_loss = self.list_of_losses[1]
        figure = plt.figure(figsize = (8,6))
        plt.plot(clustering_loss, label = 'Clustering Loss', color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Clustering Loss')
        plt.title('Plot of Clustering Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Clustering Loss.png', dpi = 300)
        plt.show()

    def plot_total_loss(self, color = 'orange'):

        total_loss = self.list_of_losses[2]
        figure = plt.figure(figsize = (8,6))
        plt.plot(total_loss, label = 'Total Loss', color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Total Loss')
        plt.title('Plot of Total Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Total Loss.png', dpi = 300)
        plt.show()

    def plot_all_losses(self, color_one = 'blue', color_two = 'green', color_three = 'orange'):

        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)

        ax1.plot(self.list_of_losses[0], label = 'Network Loss', color = color_one)
        ax2.plot(self.list_of_losses[1], label = 'Clustering Loss', color = color_two)
        ax3.plot(self.list_of_losses[2], label = 'Total Loss', color = color_three)

        ax1.legend()
        ax1.grid(b = True, which = 'major', axis = 'both')
        ax1.set_title('Network Loss')
        ax1.set_xlabel('Number of Epochs')
        ax1.set_ylabel('Network Loss')

        ax2.legend()
        ax2.grid(b = True, which = 'major', axis = 'both')
        ax2.set_title('Clustering Loss')
        ax2.set_xlabel('Number of Epochs')
        ax2.set_ylabel('Clustering Loss')

        ax3.legend()
        ax3.grid(b = True, which = 'major', axis = 'both')
        ax3.set_title('Total Loss')
        ax3.set_xlabel('Number of Epochs')
        ax3.set_ylabel('Total Loss')

        plt.tight_layout()
        plt.savefig('Combined Loss Plot.png', dpi = 300)
        plt.show()





    def apply_TSNE(self, n_components = 2, perplexity = 30.0):

        tsne_channels = []

        for i in range(self.embeddings.shape[1]):
            tsne = sklearn.manifold.TSNE(n_components = n_components, perplexity = perplexity)
            channel = self.embeddings[:,i,:].numpy()
            tsne_channel = tsne.fit_transform(channel)
            tsne_channels.append(tsne_channel)

        return tsne_channels


    def plot_bottleneck(self, tsne_channels):
        for i in range(len(tsne_channels)):
            channel = tsne_channels[i]
            label = self.labels_pred[i]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(channel[label == 0,0],
                       channel[label == 0, 1], s=1, c = 'green', label = 'Cluster 1')
            ax.scatter(channel[label == 1, 0],
                       channel[label == 1, 1], s=1, c = 'red', label = 'Cluster 2')

            plt.grid(b = True, which = 'major', axis = 'both')
            plt.title('Activation Map {}'.format(i+1))
            plt.legend()
            plt.savefig('TSNE Plot Channel {}.png'.format(i+1), dpi = 300)
            plt.show()


    def plot_nmi(self, alpha  = 0.8, color = 'indigo'):

        top_nmis = []

        for i in self.top_nmi_maps_idx:

            top_nmis.append(self.list_of_nmi[i])


        figure = plt.figure(figsize = (8,6))
        width = 0.35
        plt.bar(self.top_nmi_maps_idx, top_nmis, align = 'center', alpha = alpha, label = 'NMI', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('NMI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Channel wise NMI')
        plt.legend()
        plt.savefig('NMI.png', dpi = 300)
        plt.show()

    def plot_ari(self, list_of_ari, alpha  = 0.8, color = 'indigo'):

        figure = plt.figure(figsize = (8,6))
        y_pos = np.arange(1,len(list_of_ari)+1)
        width = 0.35
        plt.bar(y_pos, list_of_ari, align = 'center', alpha = alpha, label = 'ARI', color = color)
        plt.xlabel('Channels')
        plt.ylabel('ARI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Channel wise ARI')
        plt.legend()
        plt.savefig('ARI.png', dpi = 300)
        plt.show()

    def plot_acc(self, list_of_acc, alpha  = 0.8, color = 'indigo'):

        figure = plt.figure(figsize = (8,6))
        y_pos = np.arange(1,len(list_of_acc)+1)
        width = 0.35
        plt.bar(y_pos, list_of_acc, align = 'center', alpha = alpha, label = 'Accuracy', color = color)
        plt.xlabel('Channels')
        plt.ylabel('Accuracy')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Channel wise Accuracy')
        plt.legend()
        plt.savefig('Accuracy.png', dpi = 300)
        plt.show()


    def a_new_function_to_commit():
        pass
        
